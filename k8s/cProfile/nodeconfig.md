(base) guobuzai@Guos-MacBook-Pro cProfile % kubectl describe node hsrn-edbsc-12wvpl
Name:               hsrn-edbsc-12wvpl
Roles:              <none>
Labels:             beta.kubernetes.io/arch=amd64
                    beta.kubernetes.io/os=linux
                    cpu-feature.node.kubevirt.io/abm=true
                    cpu-feature.node.kubevirt.io/aes=true
                    cpu-feature.node.kubevirt.io/amd-ssbd=true
                    cpu-feature.node.kubevirt.io/amd-stibp=true
                    cpu-feature.node.kubevirt.io/apic=true
                    cpu-feature.node.kubevirt.io/arat=true
                    cpu-feature.node.kubevirt.io/arch-capabilities=true
                    cpu-feature.node.kubevirt.io/avx=true
                    cpu-feature.node.kubevirt.io/avx2=true
                    cpu-feature.node.kubevirt.io/bmi1=true
                    cpu-feature.node.kubevirt.io/bmi2=true
                    cpu-feature.node.kubevirt.io/clflush=true
                    cpu-feature.node.kubevirt.io/cmov=true
                    cpu-feature.node.kubevirt.io/cx16=true
                    cpu-feature.node.kubevirt.io/cx8=true
                    cpu-feature.node.kubevirt.io/de=true
                    cpu-feature.node.kubevirt.io/erms=true
                    cpu-feature.node.kubevirt.io/f16c=true
                    cpu-feature.node.kubevirt.io/fma=true
                    cpu-feature.node.kubevirt.io/fpu=true
                    cpu-feature.node.kubevirt.io/fsgsbase=true
                    cpu-feature.node.kubevirt.io/fxsr=true
                    cpu-feature.node.kubevirt.io/hypervisor=true
                    cpu-feature.node.kubevirt.io/ibpb=true
                    cpu-feature.node.kubevirt.io/ibrs=true
                    cpu-feature.node.kubevirt.io/invpcid=true
                    cpu-feature.node.kubevirt.io/invtsc=true
                    cpu-feature.node.kubevirt.io/lahf_lm=true
                    cpu-feature.node.kubevirt.io/lm=true
                    cpu-feature.node.kubevirt.io/mca=true
                    cpu-feature.node.kubevirt.io/mce=true
                    cpu-feature.node.kubevirt.io/md-clear=true
                    cpu-feature.node.kubevirt.io/mmx=true
                    cpu-feature.node.kubevirt.io/movbe=true
                    cpu-feature.node.kubevirt.io/msr=true
                    cpu-feature.node.kubevirt.io/mtrr=true
                    cpu-feature.node.kubevirt.io/nx=true
                    cpu-feature.node.kubevirt.io/pae=true
                    cpu-feature.node.kubevirt.io/pat=true
                    cpu-feature.node.kubevirt.io/pcid=true
                    cpu-feature.node.kubevirt.io/pclmuldq=true
                    cpu-feature.node.kubevirt.io/pdcm=true
                    cpu-feature.node.kubevirt.io/pdpe1gb=true
                    cpu-feature.node.kubevirt.io/pge=true
                    cpu-feature.node.kubevirt.io/pni=true
                    cpu-feature.node.kubevirt.io/popcnt=true
                    cpu-feature.node.kubevirt.io/pschange-mc-no=true
                    cpu-feature.node.kubevirt.io/pse=true
                    cpu-feature.node.kubevirt.io/pse36=true
                    cpu-feature.node.kubevirt.io/rdrand=true
                    cpu-feature.node.kubevirt.io/rdtscp=true
                    cpu-feature.node.kubevirt.io/sep=true
                    cpu-feature.node.kubevirt.io/skip-l1dfl-vmentry=true
                    cpu-feature.node.kubevirt.io/smep=true
                    cpu-feature.node.kubevirt.io/spec-ctrl=true
                    cpu-feature.node.kubevirt.io/ss=true
                    cpu-feature.node.kubevirt.io/ssbd=true
                    cpu-feature.node.kubevirt.io/sse=true
                    cpu-feature.node.kubevirt.io/sse2=true
                    cpu-feature.node.kubevirt.io/sse4.1=true
                    cpu-feature.node.kubevirt.io/sse4.2=true
                    cpu-feature.node.kubevirt.io/ssse3=true
                    cpu-feature.node.kubevirt.io/stibp=true
                    cpu-feature.node.kubevirt.io/syscall=true
                    cpu-feature.node.kubevirt.io/tsc=true
                    cpu-feature.node.kubevirt.io/tsc-deadline=true
                    cpu-feature.node.kubevirt.io/tsc_adjust=true
                    cpu-feature.node.kubevirt.io/umip=true
                    cpu-feature.node.kubevirt.io/vme=true
                    cpu-feature.node.kubevirt.io/vmx=true
                    cpu-feature.node.kubevirt.io/x2apic=true
                    cpu-feature.node.kubevirt.io/xsave=true
                    cpu-feature.node.kubevirt.io/xsaveopt=true
                    cpu-model-migration.node.kubevirt.io/Haswell-noTSX=true
                    cpu-model-migration.node.kubevirt.io/Haswell-noTSX-IBRS=true
                    cpu-model-migration.node.kubevirt.io/IvyBridge=true
                    cpu-model-migration.node.kubevirt.io/IvyBridge-IBRS=true
                    cpu-model-migration.node.kubevirt.io/Nehalem=true
                    cpu-model-migration.node.kubevirt.io/Nehalem-IBRS=true
                    cpu-model-migration.node.kubevirt.io/Penryn=true
                    cpu-model-migration.node.kubevirt.io/SandyBridge=true
                    cpu-model-migration.node.kubevirt.io/SandyBridge-IBRS=true
                    cpu-model-migration.node.kubevirt.io/Westmere=true
                    cpu-model-migration.node.kubevirt.io/Westmere-IBRS=true
                    cpu-model.node.kubevirt.io/Haswell-noTSX=true
                    cpu-model.node.kubevirt.io/Haswell-noTSX-IBRS=true
                    cpu-model.node.kubevirt.io/IvyBridge=true
                    cpu-model.node.kubevirt.io/IvyBridge-IBRS=true
                    cpu-model.node.kubevirt.io/Nehalem=true
                    cpu-model.node.kubevirt.io/Nehalem-IBRS=true
                    cpu-model.node.kubevirt.io/Penryn=true
                    cpu-model.node.kubevirt.io/SandyBridge=true
                    cpu-model.node.kubevirt.io/SandyBridge-IBRS=true
                    cpu-model.node.kubevirt.io/Westmere=true
                    cpu-model.node.kubevirt.io/Westmere-IBRS=true
                    cpu-timer.node.kubevirt.io/tsc-frequency=2599981000
                    cpu-timer.node.kubevirt.io/tsc-scalable=false
                    cpu-vendor.node.kubevirt.io/Intel=true
                    cpumanager=false
                    feature.node.kubernetes.io/cpu-cpuid.AESNI=true
                    feature.node.kubernetes.io/cpu-cpuid.AVX=true
                    feature.node.kubernetes.io/cpu-cpuid.AVX2=true
                    feature.node.kubernetes.io/cpu-cpuid.CMPXCHG8=true
                    feature.node.kubernetes.io/cpu-cpuid.FLUSH_L1D=true
                    feature.node.kubernetes.io/cpu-cpuid.FMA3=true
                    feature.node.kubernetes.io/cpu-cpuid.FXSR=true
                    feature.node.kubernetes.io/cpu-cpuid.FXSROPT=true
                    feature.node.kubernetes.io/cpu-cpuid.IBPB=true
                    feature.node.kubernetes.io/cpu-cpuid.LAHF=true
                    feature.node.kubernetes.io/cpu-cpuid.MD_CLEAR=true
                    feature.node.kubernetes.io/cpu-cpuid.MOVBE=true
                    feature.node.kubernetes.io/cpu-cpuid.OSXSAVE=true
                    feature.node.kubernetes.io/cpu-cpuid.SPEC_CTRL_SSBD=true
                    feature.node.kubernetes.io/cpu-cpuid.STIBP=true
                    feature.node.kubernetes.io/cpu-cpuid.SYSCALL=true
                    feature.node.kubernetes.io/cpu-cpuid.SYSEE=true
                    feature.node.kubernetes.io/cpu-cpuid.VMX=true
                    feature.node.kubernetes.io/cpu-cpuid.X87=true
                    feature.node.kubernetes.io/cpu-cpuid.XSAVE=true
                    feature.node.kubernetes.io/cpu-cpuid.XSAVEOPT=true
                    feature.node.kubernetes.io/cpu-cstate.enabled=true
                    feature.node.kubernetes.io/cpu-hardware_multithreading=true
                    feature.node.kubernetes.io/cpu-model.family=6
                    feature.node.kubernetes.io/cpu-model.id=63
                    feature.node.kubernetes.io/cpu-model.vendor_id=Intel
                    feature.node.kubernetes.io/cpu-pstate.status=passive
                    feature.node.kubernetes.io/cpu-pstate.turbo=true
                    feature.node.kubernetes.io/cpu-rdt.RDTCMT=true
                    feature.node.kubernetes.io/cpu-rdt.RDTMON=true
                    feature.node.kubernetes.io/custom-rdma.capable=true
                    feature.node.kubernetes.io/kernel-config.NO_HZ=true
                    feature.node.kubernetes.io/kernel-config.NO_HZ_IDLE=true
                    feature.node.kubernetes.io/kernel-version.full=5.15.0-112-generic
                    feature.node.kubernetes.io/kernel-version.major=5
                    feature.node.kubernetes.io/kernel-version.minor=15
                    feature.node.kubernetes.io/kernel-version.revision=0
                    feature.node.kubernetes.io/memory-numa=true
                    feature.node.kubernetes.io/pci-102b.present=true
                    feature.node.kubernetes.io/storage-nonrotationaldisk=true
                    feature.node.kubernetes.io/system-os_release.ID=ubuntu
                    feature.node.kubernetes.io/system-os_release.VERSION_ID=22.04
                    feature.node.kubernetes.io/system-os_release.VERSION_ID.major=22
                    feature.node.kubernetes.io/system-os_release.VERSION_ID.minor=04
                    host-model-cpu.node.kubevirt.io/Haswell-noTSX-IBRS=true
                    host-model-required-features.node.kubevirt.io/abm=true
                    host-model-required-features.node.kubevirt.io/amd-ssbd=true
                    host-model-required-features.node.kubevirt.io/amd-stibp=true
                    host-model-required-features.node.kubevirt.io/arat=true
                    host-model-required-features.node.kubevirt.io/arch-capabilities=true
                    host-model-required-features.node.kubevirt.io/f16c=true
                    host-model-required-features.node.kubevirt.io/hypervisor=true
                    host-model-required-features.node.kubevirt.io/ibpb=true
                    host-model-required-features.node.kubevirt.io/ibrs=true
                    host-model-required-features.node.kubevirt.io/invtsc=true
                    host-model-required-features.node.kubevirt.io/md-clear=true
                    host-model-required-features.node.kubevirt.io/pdcm=true
                    host-model-required-features.node.kubevirt.io/pdpe1gb=true
                    host-model-required-features.node.kubevirt.io/pschange-mc-no=true
                    host-model-required-features.node.kubevirt.io/rdrand=true
                    host-model-required-features.node.kubevirt.io/skip-l1dfl-vmentry=true
                    host-model-required-features.node.kubevirt.io/ss=true
                    host-model-required-features.node.kubevirt.io/ssbd=true
                    host-model-required-features.node.kubevirt.io/stibp=true
                    host-model-required-features.node.kubevirt.io/tsc_adjust=true
                    host-model-required-features.node.kubevirt.io/umip=true
                    host-model-required-features.node.kubevirt.io/vme=true
                    host-model-required-features.node.kubevirt.io/vmx=true
                    host-model-required-features.node.kubevirt.io/xsaveopt=true
                    hpc.nyu.edu/kvm=true
                    hpc.nyu.edu/os-version=ubuntu-22.04
                    hyperv.node.kubevirt.io/base=true
                    hyperv.node.kubevirt.io/frequencies=true
                    hyperv.node.kubevirt.io/ipi=true
                    hyperv.node.kubevirt.io/reenlightenment=true
                    hyperv.node.kubevirt.io/reset=true
                    hyperv.node.kubevirt.io/runtime=true
                    hyperv.node.kubevirt.io/synic=true
                    hyperv.node.kubevirt.io/synic2=true
                    hyperv.node.kubevirt.io/synictimer=true
                    hyperv.node.kubevirt.io/time=true
                    hyperv.node.kubevirt.io/tlbflush=true
                    hyperv.node.kubevirt.io/vpindex=true
                    katacontainers.io/kata-runtime=true
                    kubernetes.io/arch=amd64
                    kubernetes.io/hostname=hsrn-edbsc-12wvpl
                    kubernetes.io/os=linux
                    kubevirt.io/ksm-enabled=false
                    kubevirt.io/schedulable=true
                    scheduling.node.kubevirt.io/tsc-frequency-2599981000=true
                    topology.kubernetes.io/region=us-east
                    topology.kubernetes.io/zone=12wvpl
Annotations:        csi.volume.kubernetes.io/nodeid: {"cephfs.csi.ceph.com":"hsrn-edbsc-12wvpl","rbd.csi.ceph.com":"hsrn-edbsc-12wvpl"}
                    kubeadm.alpha.kubernetes.io/cri-socket: unix:///var/run/containerd/containerd.sock
                    kubevirt.io/heartbeat: 2024-06-30T02:38:34Z
                    kubevirt.io/ksm-handler-managed: false
                    nfd.node.kubernetes.io/extended-resources: 
                    nfd.node.kubernetes.io/feature-labels:
                      cpu-cpuid.AESNI,cpu-cpuid.AVX,cpu-cpuid.AVX2,cpu-cpuid.CMPXCHG8,cpu-cpuid.FLUSH_L1D,cpu-cpuid.FMA3,cpu-cpuid.FXSR,cpu-cpuid.FXSROPT,cpu-cp...
                    nfd.node.kubernetes.io/worker.version: v0.13.6
                    node.alpha.kubernetes.io/ttl: 0
                    projectcalico.org/IPv4Address: 10.33.69.9/24
                    projectcalico.org/IPv4IPIPTunnelAddr: 10.0.101.0
                    volumes.kubernetes.io/controller-managed-attach-detach: true
CreationTimestamp:  Sat, 07 Oct 2023 16:27:09 -0400
Taints:             <none>
Unschedulable:      false
Lease:              Failed to get lease: leases.coordination.k8s.io "hsrn-edbsc-12wvpl" is forbidden: User "oidc:oauth2|CILogon|http://cilogon.org/serverE/users/122265" cannot get resource "leases" in API group "coordination.k8s.io" in the namespace "kube-node-lease"
Conditions:
  Type                 Status  LastHeartbeatTime                 LastTransitionTime                Reason                       Message
  ----                 ------  -----------------                 ------------------                ------                       -------
  NetworkUnavailable   False   Wed, 12 Jun 2024 10:37:14 -0400   Wed, 12 Jun 2024 10:37:14 -0400   CalicoIsUp                   Calico is running on this node
  MemoryPressure       False   Sat, 29 Jun 2024 22:38:45 -0400   Sun, 09 Jun 2024 02:54:23 -0400   KubeletHasSufficientMemory   kubelet has sufficient memory available
  DiskPressure         False   Sat, 29 Jun 2024 22:38:45 -0400   Sun, 09 Jun 2024 02:54:23 -0400   KubeletHasNoDiskPressure     kubelet has no disk pressure
  PIDPressure          False   Sat, 29 Jun 2024 22:38:45 -0400   Sun, 09 Jun 2024 02:54:23 -0400   KubeletHasSufficientPID      kubelet has sufficient PID available
  Ready                True    Sat, 29 Jun 2024 22:38:45 -0400   Sun, 09 Jun 2024 02:54:23 -0400   KubeletReady                 kubelet is posting ready status. AppArmor enabled
Addresses:
  InternalIP:  10.33.69.9
  Hostname:    hsrn-edbsc-12wvpl
Capacity:
  cpu:                            32
  devices.kubevirt.io/kvm:        1k
  devices.kubevirt.io/tun:        1k
  devices.kubevirt.io/vhost-net:  1k
  ephemeral-storage:              1918572936Ki
  hugepages-1Gi:                  0
  hugepages-2Mi:                  0
  memory:                         396006520Ki
  pods:                           110
Allocatable:
  cpu:                            32
  devices.kubevirt.io/kvm:        1k
  devices.kubevirt.io/tun:        1k
  devices.kubevirt.io/vhost-net:  1k
  ephemeral-storage:              1768156814891
  hugepages-1Gi:                  0
  hugepages-2Mi:                  0
  memory:                         395904120Ki
  pods:                           110
System Info:
  Machine ID:                 8e690aedf95240df8a7771b503670d83
  System UUID:                4c4c4544-0054-5710-8035-b4c04f533532
  Boot ID:                    b4da69c9-cb15-46cd-8f36-8ccf1c70d39b
  Kernel Version:             5.15.0-112-generic
  OS Image:                   Ubuntu 22.04.4 LTS
  Operating System:           linux
  Architecture:               amd64
  Container Runtime Version:  containerd://1.7.2
  Kubelet Version:            v1.28.10
  Kube-Proxy Version:         v1.28.10
PodCIDR:                      10.0.15.0/24
PodCIDRs:                     10.0.15.0/24
Pods:                         not authorized
Events:                       <none>
(base) guobuzai@Guos-MacBook-Pro cProfile % 
