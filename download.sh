!/bin/bash

!gdown --id '1BzedlECiMt4n0Uc_s-jjbMegzpwsEOGq' --output model.zip
!unzip model.zip

while read url; do
    wget $url
done < urls.txt
