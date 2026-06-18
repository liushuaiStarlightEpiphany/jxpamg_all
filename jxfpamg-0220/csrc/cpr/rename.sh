for file in jx_*; do
    mv "$file" "jxf_${file#jx_}"
done