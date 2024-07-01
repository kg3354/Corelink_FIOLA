docker buildx build --platform linux/amd64 -t registry.hsrn.nyu.edu/vip/corelink-examples/fiola-process-ns . --push 

docker buildx build --platform linux/amd64 -t registry.hsrn.nyu.edu/vip/corelink-examples/fiola-process . --push 



kubectl cp fiola-process-job-r2vjd:/usr/src/app/profile_output.prof ./profile_output.prof


python -m pstats profile_output.prof

