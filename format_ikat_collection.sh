echo "Enter the absolute path of the ikat collection file (ex: /home/user/TREC-iKAT/...)"
read abs_path

sed -i 's/ClueWeb22-ID/id/' $abs_path
sed -i 's/Clean-Text/contents/' $abs_path
sed -i 's/"}/"},/ $abs_path' $abs_path

echo ']' >> $abs_path
sed -i '1s/^/[ /' $abs_path

