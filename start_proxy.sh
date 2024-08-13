sudo rm -rf dv_dbg*
DV_TGT_ROOT="/opt/sdk_ara1"
echo "${DV_TGT_ROOT}"
sudo pkill dvinfproxy
sudo rm -rf /var/run/dvproxy.pid
sudo rm -rf /dev/shm/nnapp.shm
sudo ${DV_TGT_ROOT}/dvproxy/x86_rel/dvinfproxy -f ${DV_TGT_ROOT}/dvproxy/x86_rel/firmware -S 800 