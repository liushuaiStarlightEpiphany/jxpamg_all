//========================================================================//
//  JXFPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jxf_par_csr_assumed_part.c -- assumed partition operations for parallel csr matrices.
 *  Date: 2025/10/08
 */ 
/*----------------------------------------------------
 * Functions for the IJ assumed partition
 * (Some of these were formerly in new_commpkg.c)
 *  AHB 4/06
 *-----------------------------------------------------*/

#include "jxf_mv.h"

/* This is used only in the function below */
#define CONTACT(a, b) (contact_list[(a) * 3 + (b)])

/*--------------------------------------------------------------------
 * jxf_LocateAssumedPartition
 * Reconcile assumed partition with actual partition.  Essentially
 * each processor ends of with a partition of its assumed partition.
 *--------------------------------------------------------------------*/
JXF_Int jxf_LocateAssumedPartition(MPI_Comm comm, JXF_BigInt row_start, JXF_BigInt row_end, JXF_BigInt global_first_row,
                                         JXF_BigInt global_num_rows, jxf_IJAssumedPart* part, JXF_Int myid)
{
    JXF_Int i;

    JXF_BigInt* contact_list;
    JXF_Int     contact_list_length, contact_list_storage;

    JXF_BigInt contact_row_start[2], contact_row_end[2], contact_ranges;
    JXF_Int    owner_start, owner_end;
    JXF_BigInt tmp_row_start, tmp_row_end;
    JXF_Int    complete;

    /*JXF_Int        locate_row_start[2]; */
    /*JXF_Int        locate_ranges;*/

    JXF_Int locate_row_count, rows_found;

    JXF_BigInt  tmp_range[2];
    JXF_BigInt* sortme;
    JXF_Int*    si;

    const JXF_Int flag1 = 17;

    MPI_Request* requests;
    MPI_Status   status0, *statuses;

    ;

    /*-----------------------------------------------------------
     *  Contact ranges -
     *  which rows do I have that others are assumed responsible for?
     *  (at most two ranges - maybe none)
     *-----------------------------------------------------------*/
    contact_row_start[0] = 0;
    contact_row_end[0]   = 0;
    contact_row_start[1] = 0;
    contact_row_end[1]   = 0;
    contact_ranges       = 0;

    if (row_start <= row_end) {
        /*must own at least one row*/
        if (part->row_end < row_start || row_end < part->row_start) {
            /*no overlap - so all of my rows and only one range*/
            contact_row_start[0] = row_start;
            contact_row_end[0]   = row_end;
            contact_ranges++;
        } else /* the two regions overlap - so one or two ranges */
        {
            /* check for contact rows on the low end of the local range */
            if (row_start < part->row_start) {
                contact_row_start[0] = row_start;
                contact_row_end[0]   = part->row_start - 1;
                contact_ranges++;
            }
            if (part->row_end < row_end) /* check the high end */
            {
                if (contact_ranges) /* already found one range */
                {
                    contact_row_start[1] = part->row_end + 1;
                    contact_row_end[1]   = row_end;
                } else {
                    contact_row_start[0] = part->row_end + 1;
                    contact_row_end[0]   = row_end;
                }
                contact_ranges++;
            }
        }
    }

    /*-----------------------------------------------------------
     *  Contact: find out who is assumed responsible for these
     *       ranges of contact rows and contact them
     *
     *-----------------------------------------------------------*/

    contact_list_length  = 0;
    contact_list_storage = 5;
    contact_list         = jxf_TAlloc(JXF_BigInt, contact_list_storage * 3); /*each contact needs 3 ints */

    for (i = 0; i < contact_ranges; i++) {

        /*get start and end row owners */
        jxf_GetAssumedPartitionProcFromRow(comm, contact_row_start[i], global_first_row, global_num_rows, &owner_start);
        jxf_GetAssumedPartitionProcFromRow(comm, contact_row_end[i], global_first_row, global_num_rows, &owner_end);

        if (owner_start == owner_end) /* same processor owns the whole range */
        {

            if (contact_list_length == contact_list_storage) {
                /*allocate more space*/
                contact_list_storage += 5;
                contact_list = jxf_TReAlloc(contact_list, JXF_BigInt, (contact_list_storage * 3));
            }
            CONTACT(contact_list_length, 0) = (JXF_BigInt)owner_start; /*proc #*/
            CONTACT(contact_list_length, 1) = contact_row_start[i];       /* start row */
            CONTACT(contact_list_length, 2) = contact_row_end[i];         /*end row */
            contact_list_length++;
        } else {
            complete = 0;
            while (!complete) {
                jxf_GetAssumedPartitionRowRange(comm, owner_start, global_first_row, global_num_rows, &tmp_row_start, &tmp_row_end);

                if (tmp_row_end >= contact_row_end[i]) {
                    tmp_row_end = contact_row_end[i];
                    complete    = 1;
                }
                if (tmp_row_start < contact_row_start[i]) {
                    tmp_row_start = contact_row_start[i];
                }

                if (contact_list_length == contact_list_storage) {
                    /*allocate more space*/
                    contact_list_storage += 5;
                    contact_list = jxf_TReAlloc(contact_list, JXF_BigInt, (contact_list_storage * 3));
                }

                CONTACT(contact_list_length, 0) = (JXF_BigInt)owner_start; /*proc #*/
                CONTACT(contact_list_length, 1) = tmp_row_start;              /* start row */
                CONTACT(contact_list_length, 2) = tmp_row_end;                /*end row */
                contact_list_length++;
                owner_start++; /*processors are seqential */
            }
        }
    }

    requests = jxf_CTAlloc(MPI_Request, contact_list_length);
    statuses = jxf_CTAlloc(MPI_Status, contact_list_length);

    /*send out messages */
    for (i = 0; i < contact_list_length; i++) {
        jxf_MPI_Isend(&CONTACT(i, 1), 2, JXF_MPI_INT, CONTACT(i, 0), flag1, comm, &requests[i]);
        /*jxf_MPI_COMM_WORLD, &requests[i]);*/
    }

    /*-----------------------------------------------------------
     *  Locate ranges -
     *  which rows in my assumed range do I not own
     *  (at most two ranges - maybe none)
     *  locate_row_count = total number of rows I must locate
     *-----------------------------------------------------------*/

    locate_row_count = 0;

    /*locate_row_start[0]=0;
    locate_row_start[1]=0;*/

    /*locate_ranges = 0;*/

    if (part->row_end < row_start || row_end < part->row_start)
    /*no overlap - so all of my assumed rows */
    {
        /*locate_row_start[0] = part->row_start;*/
        /*locate_ranges++;*/
        locate_row_count += part->row_end - part->row_start + 1;
    } else /* the two regions overlap */
    {
        if (part->row_start < row_start) {
            /* check for locate rows on the low end of the local range */
            /*locate_row_start[0] = part->row_start;*/
            /*locate_ranges++;*/
            locate_row_count += (row_start - 1) - part->row_start + 1;
        }
        if (row_end < part->row_end) /* check the high end */
        {
            /*if (locate_ranges)*/ /* already have one range */
            /*{
              locate_row_start[1] = row_end +1;
              }
              else
              {
              locate_row_start[0] = row_end +1;
              }*/
            /*locate_ranges++;*/
            locate_row_count += part->row_end - (row_end + 1) + 1;
        }
    }

    /*-----------------------------------------------------------
     * Receive messages from other procs telling us where
     * all our  locate rows actually reside
     *-----------------------------------------------------------*/

    /* we will keep a partition of our assumed partition - list ourselves
       first.  We will sort later with an additional index.
       In practice, this should only contain a few processors */

    /*which part do I own?*/
    tmp_row_start = jxf_max(part->row_start, row_start);
    tmp_row_end   = jxf_min(row_end, part->row_end);

    if (tmp_row_start <= tmp_row_end) {
        part->proc_list[0]      = myid;
        part->row_start_list[0] = tmp_row_start;
        part->row_end_list[0]   = tmp_row_end;
        part->length++;
    }

    /* now look for messages that tell us which processor has our locate rows */
    /* these will be blocking receives as we know how many to expect and they should
        be waiting (and we don't want to continue on without them) */

    rows_found = 0;

    while (rows_found != locate_row_count) {
        jxf_MPI_Recv(tmp_range, 2, JXF_MPI_INT, MPI_ANY_SOURCE, flag1, comm, &status0);
        /*flag1 , jxf_MPI_COMM_WORLD, &status0);*/

        if (part->length == part->storage_length) {
            part->storage_length += 10;
            part->proc_list      = jxf_TReAlloc(part->proc_list, JXF_Int, part->storage_length);
            part->row_start_list = jxf_TReAlloc(part->row_start_list, JXF_BigInt, part->storage_length);
            part->row_end_list   = jxf_TReAlloc(part->row_end_list, JXF_BigInt, part->storage_length);
        }
        part->row_start_list[part->length] = tmp_range[0];
        part->row_end_list[part->length]   = tmp_range[1];

        part->proc_list[part->length] = status0.MPI_SOURCE;
        rows_found += tmp_range[1] - tmp_range[0] + 1;

        part->length++;
    }

    /*In case the partition of the assumed partition is longish,
      we would like to know the sorted order */
    si     = jxf_CTAlloc(JXF_Int, part->length);
    sortme = jxf_CTAlloc(JXF_BigInt, part->length);

    for (i = 0; i < part->length; i++) {
        si[i]     = i;
        sortme[i] = part->row_start_list[i];
    }
    jxf_qsort1(sortme, si, 0, (part->length) - 1);
    part->sort_index = si;

    /*free the requests */
    jxf_MPI_Waitall(contact_list_length, requests, statuses);

    jxf_TFree(statuses);
    jxf_TFree(requests);

    jxf_TFree(sortme);
    jxf_TFree(contact_list);

    return jxf_error_flag;
}

jxf_IJAssumedPart* jxf_AssumedPartitionCreate(MPI_Comm comm, JXF_BigInt global_num, JXF_BigInt start, JXF_BigInt end)
{
    jxf_IJAssumedPart* apart;
    JXF_Int            myid;

    jxf_MPI_Comm_rank(comm, &myid);

    /* allocate space */
    apart = jxf_CTAlloc(jxf_IJAssumedPart, 1);

    jxf_GetAssumedPartitionRowRange(comm, myid, 0, global_num, &(apart->row_start), &(apart->row_end));

    /*allocate some space for the partition of the assumed partition */
    apart->length = 0;
    /*room for 10 owners of the assumed partition*/
    apart->storage_length = 10; /*need to be >=1 */
    apart->proc_list      = jxf_TAlloc(JXF_Int, apart->storage_length);
    apart->row_start_list = jxf_TAlloc(JXF_BigInt, apart->storage_length);
    apart->row_end_list   = jxf_TAlloc(JXF_BigInt, apart->storage_length);

    /* now we want to reconcile our actual partition with the assumed partition */
    jxf_LocateAssumedPartition(comm, start, end, 0, global_num, apart, myid);

    return apart;
}

// /*--------------------------------------------------------------------
//  * jxf_ParCSRMatrixCreateAssumedPartition -
//  * Each proc gets it own range. Then
//  * each needs to reconcile its actual range with its assumed
//  * range - the result is essentila a partition of its assumed range -
//  * this is the assumed partition.
//  *--------------------------------------------------------------------*/
// JXF_Int jxf_ParCSRMatrixCreateAssumedPartition(jxf_ParCSRMatrix* matrix)
// {
//     JXF_BigInt global_num_cols;
//     /* JXF_Int myid; */
//     JXF_BigInt row_start = 0, row_end = 0, col_start = 0, col_end = 0;

//     MPI_Comm comm;

//     jxf_IJAssumedPart* apart;

//     global_num_cols = jxf_ParCSRMatrixGlobalNumCols(matrix);
//     comm            = jxf_ParCSRMatrixComm(matrix);

//     /* find out my actualy range of rows and columns */
//     jxf_ParCSRMatrixGetLocalRange(matrix, &row_start, &row_end, /* these two are not used */
//                                      &col_start, &col_end);
//     /* get my assumed partitioning  - we want partitioning of the vector that the
//        matrix multiplies - so we use the col start and end */
//     apart = jxf_AssumedPartitionCreate(comm, global_num_cols, col_start, col_end);

//     /* this partition will be saved in the matrix data structure until the matrix is destroyed */
//     jxf_ParCSRMatrixAssumedPartition(matrix) = apart;

//     return jxf_error_flag;
// }

/*--------------------------------------------------------------------
 * jxf_AssumedPartitionDestroy
 *--------------------------------------------------------------------*/
JXF_Int jxf_AssumedPartitionDestroy(jxf_IJAssumedPart* apart)
{
    if (apart->storage_length > 0) {
        jxf_TFree(apart->proc_list);
        jxf_TFree(apart->row_start_list);
        jxf_TFree(apart->row_end_list);
        jxf_TFree(apart->sort_index);
    }

    jxf_TFree(apart);

    return jxf_error_flag;
}

// /*--------------------------------------------------------------------
//  * jxf_GetAssumedPartitionProcFromRow
//  * Assumed partition for IJ case. Given a particular row j, return
//  * the processor that is assumed to own that row.
//  *--------------------------------------------------------------------*/

// JXF_Int jxf_GetAssumedPartitionProcFromRow(MPI_Comm comm, JXF_BigInt row, JXF_BigInt global_first_row, JXF_BigInt global_num_rows,
//                                                  JXF_Int* proc_id)
// {
//     JXF_Int    num_procs;
//     JXF_BigInt size, switch_row, extra;

//     jxf_MPI_Comm_size(comm, &num_procs);
//     /*jxf_MPI_Comm_size(jxf_MPI_COMM_WORLD, &num_procs );*/

//     /* j = floor[(row*p/N]  - this overflows*/
//     /* *proc_id = (row*num_procs)/global_num_rows;*/

//     /* this looks a bit odd, but we have to be very careful that
//        this function and the next are inverses - and rounding
//        errors make this difficult!!!!! */

//     size       = global_num_rows / (JXF_BigInt)num_procs;
//     extra      = global_num_rows - size * (JXF_BigInt)num_procs;
//     switch_row = global_first_row + (size + 1) * extra;

//     if (row >= switch_row) {
//         *proc_id = (JXF_Int)(extra + (row - switch_row) / size);
//     } else {
//         *proc_id = (JXF_Int)((row - global_first_row) / (size + 1));
//     }

//     return jxf_error_flag;
// }

// /*--------------------------------------------------------------------
//  * jxf_GetAssumedPartitionRowRange
//  * Assumed partition for IJ case. Given a particular processor id, return
//  * the assumed range of rows ([row_start, row_end]) for that processor.
//  *--------------------------------------------------------------------*/

// JXF_Int jxf_GetAssumedPartitionRowRange(MPI_Comm comm, JXF_Int proc_id, JXF_BigInt global_first_row, JXF_BigInt global_num_rows,
//                                               JXF_BigInt* row_start, JXF_BigInt* row_end)
// {
//     JXF_Int    num_procs;
//     JXF_Int    extra;
//     JXF_BigInt size;

//     jxf_MPI_Comm_size(comm, &num_procs);
//     /*jxf_MPI_Comm_size(jxf_MPI_COMM_WORLD, &num_procs );*/

//     /* this may look non-intuitive, but we have to be very careful that
//         this function and the next are inverses - and avoiding overflow and
//         rounding errors makes this difficult! */

//     size  = global_num_rows / (JXF_BigInt)num_procs;
//     extra = (JXF_Int)(global_num_rows - size * (JXF_BigInt)num_procs);

//     *row_start = global_first_row + size * (JXF_BigInt)proc_id;
//     *row_start += (JXF_BigInt)jxf_min(proc_id, extra);

//     *row_end = global_first_row + size * (JXF_BigInt)(proc_id + 1);
//     *row_end += (JXF_BigInt)jxf_min(proc_id + 1, extra);
//     *row_end = *row_end - 1;

//     return jxf_error_flag;
// }

/*--------------------------------------------------------------------
 * jxf_ParVectorCreateAssumedPartition -

 * Essentially the same as for a matrix!

 * Each proc gets it own range. Then
 * each needs to reconcile its actual range with its assumed
 * range - the result is essentila a partition of its assumed range -
 * this is the assumed partition.
 *--------------------------------------------------------------------*/

JXF_Int jxf_ParVectorCreateAssumedPartition(jxf_ParVector* vector)
{
    JXF_BigInt global_num;
    JXF_Int    myid;
    JXF_BigInt start = 0, end = 0;

    MPI_Comm comm;

    jxf_IJAssumedPart* apart;

    global_num = jxf_ParVectorGlobalSize(vector);
    comm       = jxf_ParVectorComm(vector);

    /* find out my actualy range of rows */
    start = jxf_ParVectorFirstIndex(vector);
    end   = jxf_ParVectorLastIndex(vector);

    jxf_MPI_Comm_rank(comm, &myid);

    /* allocate space */
    apart = jxf_CTAlloc(jxf_IJAssumedPart, 1);

    /* get my assumed partitioning  - we want partitioning of the vector that the
       matrix multiplies - so we use the col start and end */
    jxf_GetAssumedPartitionRowRange(comm, myid, 0, global_num, &(apart->row_start), &(apart->row_end));

    /*allocate some space for the partition of the assumed partition */
    apart->length = 0;
    /*room for 10 owners of the assumed partition*/
    apart->storage_length = 10; /*need to be >=1 */
    apart->proc_list      = jxf_TAlloc(JXF_Int, apart->storage_length);
    apart->row_start_list = jxf_TAlloc(JXF_BigInt, apart->storage_length);
    apart->row_end_list   = jxf_TAlloc(JXF_BigInt, apart->storage_length);

    /* now we want to reconcile our actual partition with the assumed partition */
    jxf_LocateAssumedPartition(comm, start, end, 0, global_num, apart, myid);

    /* this partition will be saved in the vector data structure until the vector is destroyed */
    jxf_ParVectorAssumedPartition(vector) = apart;

    return jxf_error_flag;
}
