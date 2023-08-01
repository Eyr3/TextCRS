
# cd scratch place
cd /data/xinyu/data/cv/driving/
  
# Download zip dataset from Google Drive
filename='07012018.zip'
fileid='1PZWa6H0i1PCH9zuYcIh5Ouk_p-9Gh58B'
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}
rm ./cookie
  
# Unzip
unzip -q ${filename}
rm ${filename}
  
# cd out
cd