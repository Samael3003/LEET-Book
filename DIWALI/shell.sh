

# 192. Word Frequency
# Read from the file words.txt and output the word frequency list to stdout.

sed -e 's/ /\n/g' words.txt | sed -e '/^$/d' | sort | uniq -c | sort -r | awk '{print $2" "$1}'




# 193. Valid Phone Numbers
# Read from the file file.txt and output all valid phone numbers to stdout.

grep -E '^(\([0-9]{3}\) [0-9]{3}-[0-9]{4})$|^([0-9]{3}-[0-9]{3}-[0-9]{4})$' file.txt




# 194. Transpose File
# Read from the file file.txt and print its transposed content to stdout.

while read -a line; do
    for ((i=0; i < "${#line[@]}"; i++)); do
        a[$i]="${a[$i]} ${line[$i]}"
    done
done < file.txt
for ((i=0; i < ${#a[@]}; i++)); do
    echo ${a[i]}
done



# 195. Tenth Line
# Read from the file file.txt and output the tenth line to stdout.

cnt=0
while read line && [ $cnt -le 10 ]; do
  let 'cnt = cnt + 1'
  if [ $cnt -eq 10 ]; then
    echo $line
    exit 0
  fi
done < file.txt

