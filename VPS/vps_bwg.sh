#!/bin/bash -
# Install SS on CentOS 7

echo "Installing Shadowsocks..."

random-string()
{
    cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w ${1:-32} | head -n 1
}

CONFIG_FILE=/etc/shadowsocks.json
SERVICE_FILE=/etc/systemd/system/shadowsocks.service
SS_PASSWORD=$(random-string 32)
SS_PORT=10001
SS_TIMEOUT=600
SS_METHOD=aes-256-cfb
SS_IP=`ip route get 1 | awk '{print $NF;exit}'`
GET_PIP_FILE=/tmp/get-pip.py
WAITING_SEC=5 # waiting seconds

echo "install pip..."
curl "https://bootstrap.pypa.io/get-pip.py" -o "${GET_PIP_FILE}"
python ${GET_PIP_FILE}
pip install --upgrade pip

echo "install shadowsocks..."
pip install shadowsocks
pip install --upgrade shadowsocks

# create shadowsocls config file
# if multi-user are neccessary, add different <port, password> pair in "port_password"
# Note: each port can be used only once.
# Note: "port_password" must have a comma, and the last <port, password> pair should not be
#   followed by a comma
cat <<EOF | sudo tee ${CONFIG_FILE}
{
  "server": "0.0.0.0",
  "port_password": {
      "${SS_PORT}": "${SS_PASSWORD}"
  },
  "timeout": ${SS_TIMEOUT},
  "method": "${SS_METHOD}"
}
EOF

# create service
echo "write service file..."
cat <<EOF | sudo tee ${SERVICE_FILE}
[Unit]
Description=Shadowsocks

[Service]
TimeoutStartSec=0
ExecStart=/usr/bin/ssserver -c ${CONFIG_FILE}

[Install]
WantedBy=multi-user.target
EOF

# reload firewall, if multi-user is neccessary, add more ss_port
echo "set firewall port..."
firewall-cmd --reload
firewall-cmd --zone=public --add-port=${SS_PORT}/tcp --permanent
firewall-cmd --zone=public --add-port=${SS_PORT}/udp --permanent
firewall-cmd --reload
sleep ${WAITING_SEC}

# start service
echo "start shadowsocks service..."
systemctl enable shadowsocks
systemctl start shadowsocks

echo "waiting shadowsocks starting for ${WAITING_SEC} sec..."
sleep ${WAITING_SEC}
# view service status
echo "================================"
echo ""
echo "Congratulations! Shadowsocks has been installed on your system."
echo "You shadowsocks connection info:"
echo "--------------------------------"
echo "server:      ${SS_IP}"
echo "server_port: ${SS_PORT}"
echo "password:    ${SS_PASSWORD}"
echo "method:      ${SS_METHOD}"
echo "--------------------------------"
echo ""

echo "Please ensure the shadowsocks service is active!!!"
systemctl status shadowsocks -l