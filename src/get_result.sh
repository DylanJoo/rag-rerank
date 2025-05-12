data=test

K=-1
# K=10
# K=20

N=-1
# N=1024
# thres=_thres_5

# checking result time
echo 'rac evaluation time'
ls -l results/${N}/${data}-*vanilla_${K}* | cut -f6,7,8,9 -d ' '

# checking rag time
echo 'rag evaluation time'
# ls -l logs/meta-llama/Llama-3.1-70B-Instruct/${N}/${data}-*vanilla_${K}* | cut -f6,7,8,9 -d ' '
ls -l logs/meta-llama/Llama-3.1-70B-Instruct/rag_${N}${thres}/${data}-*vanilla_${K}* | cut -f6,7,8,9 -d ' '

# get RAC results
echo 'rac evaluation result'
tail logs/rac_${K}${thres}/${data}-*vanilla_${K}* | grep '##'

# checking rag time
echo 'rag evaluation result'
# tail logs/meta-llama/Llama-3.1-70B-Instruct/${N}/${data}-*vanilla_${K}* | grep '##'
tail logs/meta-llama/Llama-3.1-70B-Instruct/rag_${N}${thres}/${data}-*vanilla_${K}* | grep '##'
