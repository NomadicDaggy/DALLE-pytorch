while read -r line; do
    /usr/bin/time -p python genrank.py --num_images 512 --out_path "../../icybox/cool-frog-21-512" --dalle_path "$line" --text "this colorful bird has a yellow breast, with a black crown and a black cheek patch"
done < models-to-rank-cool-frog-21.txt