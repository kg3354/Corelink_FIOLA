// first change requirements and dockerfile to consumer

docker buildx build  -t registry.hsrn.nyu.edu/vip/corelink-examples/fiola-process.  --platform linux/amd64 --push
docker push registry.hsrn.nyu.edu/vip/corelink-examples/fiola-process

docker buildx build --platform linux/amd64 -t registry.hsrn.nyu.edu/vip/corelink-examples/fiola-process-ns . --push


// apply to k8s

kubectl apply -f fiola-process.yaml
 

// to delete:
kubectl delete job fiola-process-job

 
// to get output
// replace xxxx with actual pod name
kubectl cp kafka-consumer-job-xxxx:/usr/src/app/consumer_output.txt ./consumer_output.txt    

