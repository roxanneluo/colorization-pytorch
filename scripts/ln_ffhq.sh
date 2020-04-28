set -x

ffhq_dir="$1"

rm -rf dataset/ffhq

mkdir dataset/ffhq

for phase in train val test; do
    mkdir "dataset/ffhq/${phase}"
    ln -s "${ffhq_dir}/${phase}" "dataset/ffhq/${phase}/dummy"
done
