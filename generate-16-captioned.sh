while read -r line; do
    /usr/bin/time -p python genrank.py --num_images 512 --out_path "/media/daggy/mag/dalle-16-captions/$line" --dalle_path ./cool-frog-21/cool-frog-21-90.pt --text "$line"
done < 16-captions.txt