# import requests


# #url = 'http://google.com/favicon.ico'
# url = 'https://ir.nist.gov/cdip/cdip-images/images.a.a.cpio'
# print('Hola')
# r = requests.get(url, allow_redirects=False)
# print(r)
# #open('google.ico', 'wb').write(r.content)

import urllib.request  as urllib2
import string
import os

alphabet = list(string.ascii_lowercase)
save_path = '/media/bscuser/PEEKBOX/Tobacco/'

for letter in alphabet:
    for letter_2 in alphabet:
        #url = "https://ir.nist.gov/cdip/cdip-images/images" + "." + letter + "." + letter_2 + "." + "cpio"
        #print(url)
        if (letter != 'a' or letter_2 != 'a'):
            url = "https://ir.nist.gov/cdip/cdip-images/images" + "." + letter + "." + letter_2 + "." + "cpio"
            print(url)
            file_name = url.split('/')[-1]
            print(file_name)
            u = urllib2.urlopen(url)
            meta = u.info()
            file_dir = os.path.join(save_path,file_name)
            print(file_dir)
            f = open(file_dir, 'wb')
            print(meta)
            #file_size = int(meta.getheaders("Content-Length")[0])
            #print("Downloading: %s Bytes: %s" % (file_name, file_size))
            file_size_dl = 0
            block_sz = 8192
            while True:
                buffer = u.read(block_sz)
                if not buffer:
                    break

                file_size_dl += len(buffer)
                f.write(buffer)
                #status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
                #status = status + chr(8)*(len(status)+1)
                if file_size_dl%100000 == 0:
                    print(file_size_dl/1048576)
                    print('MB')

            f.close()