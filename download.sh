curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF" > /dev/null
CODE="$(awk '/_warning_/ {print $NF}' /tmp/cookie)"  
curl -Lb /tmp/cookie "https://drive.google.com/uc?export=download&confirm=${CODE}&id=16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF" -o <FILE_NAME>