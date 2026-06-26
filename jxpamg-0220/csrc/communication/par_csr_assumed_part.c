//========================================================================//
//  JXPAMG(IAPCM & XTU Parallel Algebraic Multigrid) (c) 2009-2013        //
//  Institute of Applied Physics and Computational Mathematics            //
//  School of Mathematics and Computational Science Xiangtan University   //
//========================================================================//

/*!
 *  jx_par_csr_assumed_part.c -- assumed partition operations for parallel csr matrices.
 *  Date: 2025/10/08
 */ 
/*----------------------------------------------------
 * Functions for the IJ assumed partition
 * (Some of these were formerly in new_commpkg.c)
 *  AHB 4/06
 *-----------------------------------------------------*/

#include "jx_mv.h"

/* This is used only in the function below */
#define CONTACT(a, b) (contact_list[(a) * 3 + (b)])
#define JX_BigInt int
/*--------------------------------------------------------------------
 * jx_LocateAssumedPartition
 * Reconcile assumed partition with actual partition.  Essentially
 * each processor ends of with a partition of its assumed partition.
 *--------------------------------------------------------------------*/
JX_Int jx_LocateAssumedPartition(MPI_Comm comm, JX_BigInt row_start, JX_BigInt row_end, JX_BigInt global_first_row,
                                         JX_BigInt global_num_rows, jx_IJAssumedPart* part, JX_Int myid)
{
    JX_Int i;

    JX_BigInt* contact_list;
    JX_Int     contact_list_length, contact_list_storage;

    JX_BigInt contact_row_start[2], contact_row_end[2], contact_ranges;
    JX_Int    owner_start, owner_end;
    JX_BigInt tmp_row_start, tmp_row_end;
    JX_Int    complete;

    /*JX_Int        locate_row_start[2]; */
    /*JX_Int        locate_ranges;*/

    JX_Int locate_row_count, rows_found;

    JX_BigInt  tmp_range[2];
    JX_BigInt* sortme;
    JX_Int*    si;

    const JX_Int flag1 = 17;

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
    contact_list         = jx_TAlloc(JX_BigInt, contact_list_storage * 3); /*each contact needs 3 ints */

    for (i = 0; i < contact_ranges; i++) {

        /*get start and end row owners */
        jx_GetAssumedPartitionProcFromRow(comm, contact_row_start[i], global_first_row, global_num_rows, &owner_start);
        jx_GetAssumedPartitionProcFromRow(comm, contact_row_end[i], global_first_row, global_num_rows, &owner_end);

        if (owner_start == owner_end) /* same processor owns the whole range */
        {

            if (contact_list_length == contact_list_storage) {
                /*allocate more space*/
                contact_list_storage += 5;
                contact_list = jx_TReAlloc(contact_list, JX_BigInt, (contact_list_storage * 3));
            }
            CONTACT(contact_list_length, 0) = (JX_BigInt)owner_start; /*proc #*/
            CONTACT(contact_list_length, 1) = contact_row_start[i];       /* start row */
            CONTACT(contact_list_length, 2) = contact_row_end[i];         /*end row */
            contact_list_length++;
        } else {
            complete = 0;
            while (!complete) {
                jx_GetAssumedPartitionRowRange(comm, owner_start, global_first_row, global_num_rows, &tmp_row_start, &tmp_row_end);

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
                    contact_list = jx_TReAlloc(contact_list, JX_BigInt, (contact_list_storage * 3));
                }

                CONTACT(contact_list_length, 0) = (JX_BigInt)owner_start; /*proc #*/
                CONTACT(contact_list_length, 1) = tmp_row_start;              /* start row */
                CONTACT(contact_list_length, 2) = tmp_row_end;                /*end row */
                contact_list_length++;
                owner_start++; /*processors are seqential */
            }
        }
    }

    requests = jx_CTAlloc(MPI_Request, contact_list_length);
    statuses = jx_CTAlloc(MPI_Status, contact_list_length);

    /*send out messages */
    for (i = 0; i < contact_list_length; i++) {
        jx_MPI_Isend(&CONTACT(i, 1), 2, JX_MPI_INT, CONTACT(i, 0), flag1, comm, &requests[i]);
        /*jx_MPI_COMM_WORLD, &requests[i]);*/
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
    tmp_row_start = jx_max(part->row_start, row_start);
    tmp_row_end   = jx_min(row_end, part->row_end);

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
        jx_MPI_Recv(tmp_range, 2, JX_MPI_INT, MPI_ANY_SOURCE, flag1, comm, &status0);
        /*flag1 , jx_MPI_COMM_WORLD, &status0);*/

        if (part->length == part->storage_length) {
            part->storage_length += 10;
            part->proc_list      = jx_TReAlloc(part->proc_list, JX_Int, part->storage_length);
            part->row_start_list = jx_TReAlloc(part->row_start_list, JX_BigInt, part->storage_length);
            part->row_end_list   = jx_TReAlloc(part->row_end_list, JX_BigInt, part->storage_length);
        }
        part->row_start_list[part->length] = tmp_range[0];
        part->row_end_list[part->length]   = tmp_range[1];

        part->proc_list[part->length] = status0.MPI_SOURCE;
        rows_found += tmp_range[1] - tmp_range[0] + 1;

        part->length++;
    }

    /*In case the partition of the assumed partition is longish,
      we would like to know the sorted order */
    si     = jx_CTAlloc(JX_Int, part->length);
    sortme = jx_CTAlloc(JX_BigInt, part->length);

    for (i = 0; i < part->length; i++) {
        si[i]     = i;
        sortme[i] = part->row_start_list[i];
    }
    jx_qsort1(sortme, si, 0, (part->length) - 1);
    part->sort_index = si;

    /*free the requests */
    jx_MPI_Waitall(contact_list_length, requests, statuses);

    jx_TFree(statuses);
    jx_TFree(requests);

    jx_TFree(sortme);
    jx_TFree(contact_list);

    return jx_error_flag;
}

jx_IJAssumedPart* jx_AssumedPartitionCreate(MPI_Comm comm, JX_BigInt global_num, JX_BigInt start, JX_BigInt end)
{
    jx_IJAssumedPart* apart;
    JX_Int            myid;

    jx_MPI_Comm_rank(comm, &myid);

    /* allocate space */
    apart = jx_CTAlloc(jx_IJAssumedPart, 1);

    jx_GetAssumedPartitionRowRange(comm, myid, 0, global_num, &(apart->row_start), &(apart->row_end));

    /*allocate some space for the partition of the assumed partition */
    apart->length = 0;
    /*room for 10 owners of the assumed partition*/
    apart->storage_length = 10; /*need to be >=1 */
    apart->proc_list      = jx_TAlloc(JX_Int, apart->storage_length);
    apart->row_start_list = jx_TAlloc(JX_BigInt, apart->storage_length);
    apart->row_end_list   = jx_TAlloc(JX_BigInt, apart->storage_length);

    /* now we want to reconcile our actual partition with the assumed partition */
    jx_LocateAssumedPartition(comm, start, end, 0, global_num, apart, myid);

    return apart;
}

// /*--------------------------------------------------------------------
//  * jx_ParCSRMatrixCreateAssumedPartition -
//  * Each proc gets it own range. Then
//  * each needs to reconcile its actual range with its assumed
//  * range - the result is essentila a partition of its assumed range -
//  * this is the assumed partition.
//  *--------------------------------------------------------------------*/
// JX_Int jx_ParCSRMatrixCreateAssumedPartition(jx_ParCSRMatrix* matrix)
// {
//     JX_BigInt global_num_cols;
//     /* JX_Int myid; */
//     JX_BigInt row_start = 0, row_end = 0, col_start = 0, col_end = 0;

//     MPI_Comm comm;

//     jx_IJAssumedPart* apart;

//     global_num_cols = jx_ParCSRMatrixGlobalNumCols(matrix);
//     comm            = jx_ParCSRMatrixComm(matrix);

//     /* find out my actualy range of rows and columns */
//     jx_ParCSRMatrixGetLocalRange(matrix, &row_start, &row_end, /* these two are not used */
//                                      &col_start, &col_end);
//     /* get my assumed partitioning  - we want partitioning of the vector that the
//        matrix multiplies - so we use the col start and end */
//     apart = jx_AssumedPartitionCreate(comm, global_num_cols, col_start, col_end);

//     /* this partition will be saved in the matrix data structure until the matrix is destroyed */
//     jx_ParCSRMatrixAssumedPartition(matrix) = apart;

//     return jx_error_flag;
// }

/*--------------------------------------------------------------------
 * jx_AssumedPartitionDestroy
 *--------------------------------------------------------------------*/
JX_Int jx_AssumedPartitionDestroy(jx_IJAssumedPart* apart)
{
    if (apart->storage_length > 0) {
        jx_TFree(apart->proc_list);
        jx_TFree(apart->row_start_list);
        jx_TFree(apart->row_end_list);
        jx_TFree(apart->sort_index);
    }

    jx_TFree(apart);

    return jx_error_flag;
}

// /*--------------------------------------------------------------------
//  * jx_GetAssumedPartitionProcFromRow
//  * Assumed partition for IJ case. Given a particular row j, return
//  * the processor that is assumed to own that row.
//  *--------------------------------------------------------------------*/

// JX_Int jx_GetAssumedPartitionProcFromRow(MPI_Comm comm, JX_BigInt row, JX_BigInt global_first_row, JX_BigInt global_num_rows,
//                                                  JX_Int* proc_id)
// {
//     JX_Int    num_procs;
//     JX_BigInt size, switch_row, extra;

//     jx_MPI_Comm_size(comm, &num_procs);
//     /*jx_MPI_Comm_size(jx_MPI_COMM_WORLD, &num_procs );*/

//     /* j = floor[(row*p/N]  - this overflows*/
//     /* *proc_id = (row*num_procs)/global_num_rows;*/

//     /* this looks a bit odd, but we have to be very careful that
//        this function and the next are inverses - and rounding
//        errors make this difficult!!!!! */

//     size       = global_num_rows / (JX_BigInt)num_procs;
//     extra      = global_num_rows - size * (JX_BigInt)num_procs;
//     switch_row = global_first_row + (size + 1) * extra;

//     if (row >= switch_row) {
//         *proc_id = (JX_Int)(extra + (row - switch_row) / size);
//     } else {
//         *proc_id = (JX_Int)((row - global_first_row) / (size + 1));
//     }

//     return jx_error_flag;
// }

// /*--------------------------------------------------------------------
//  * jx_GetAssumedPartitionRowRange
//  * Assumed partition for IJ case. Given a particular processor id, return
//  * the assumed range of rows ([row_start, row_end]) for that processor.
//  *--------------------------------------------------------------------*/

// JX_Int jx_GetAssumedPartitionRowRange(MPI_Comm comm, JX_Int proc_id, JX_BigInt global_first_row, JX_BigInt global_num_rows,
//                                               JX_BigInt* row_start, JX_BigInt* row_end)
// {
//     JX_Int    num_procs;
//     JX_Int    extra;
//     JX_BigInt size;

//     jx_MPI_Comm_size(comm, &num_procs);
//     /*jx_MPI_Comm_size(jx_MPI_COMM_WORLD, &num_procs );*/

//     /* this may look non-intuitive, but we have to be very careful that
//         this function and the next are inverses - and avoiding overflow and
//         rounding errors makes this difficult! */

//     size  = global_num_rows / (JX_BigInt)num_procs;
//     extra = (JX_Int)(global_num_rows - size * (JX_BigInt)num_procs);

//     *row_start = global_first_row + size * (JX_BigInt)proc_id;
//     *row_start += (JX_BigInt)jx_min(proc_id, extra);

//     *row_end = global_first_row + size * (JX_BigInt)(proc_id + 1);
//     *row_end += (JX_BigInt)jx_min(proc_id + 1, extra);
//     *row_end = *row_end - 1;

//     return jx_error_flag;
// }

/*--------------------------------------------------------------------
 * jx_ParVectorCreateAssumedPartition -

 * Essentially the same as for a matrix!

 * Each proc gets it own range. Then
 * each needs to reconcile its actual range with its assumed
 * range - the result is essentila a partition of its assumed range -
 * this is the assumed partition.
 *--------------------------------------------------------------------*/

JX_Int jx_ParVectorCreateAssumedPartition(jx_ParVector* vector)
{
    JX_BigInt global_num;
    JX_Int    myid;
    JX_BigInt start = 0, end = 0;

    MPI_Comm comm;

    jx_IJAssumedPart* apart;

    global_num = jx_ParVectorGlobalSize(vector);
    comm       = jx_ParVectorComm(vector);

    /* find out my actualy range of rows */
    start = jx_ParVectorFirstIndex(vector);
    end   = jx_ParVectorLastIndex(vector);

    jx_MPI_Comm_rank(comm, &myid);

    /* allocate space */
    apart = jx_CTAlloc(jx_IJAssumedPart, 1);

    /* get my assumed partitioning  - we want partitioning of the vector that the
       matrix multiplies - so we use the col start and end */
    jx_GetAssumedPartitionRowRange(comm, myid, 0, global_num, &(apart->row_start), &(apart->row_end));

    /*allocate some space for the partition of the assumed partition */
    apart->length = 0;
    /*room for 10 owners of the assumed partition*/
    apart->storage_length = 10; /*need to be >=1 */
    apart->proc_list      = jx_TAlloc(JX_Int, apart->storage_length);
    apart->row_start_list = jx_TAlloc(JX_BigInt, apart->storage_length);
    apart->row_end_list   = jx_TAlloc(JX_BigInt, apart->storage_length);

    /* now we want to reconcile our actual partition with the assumed partition */
    jx_LocateAssumedPartition(comm, start, end, 0, global_num, apart, myid);

    /* this partition will be saved in the vector data structure until the vector is destroyed */
    jx_ParVectorAssumedPartition(vector) = apart;

    return jx_error_flag;
}
