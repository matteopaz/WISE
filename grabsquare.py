import numpy as np
import subprocess
import time
import requests

COLS = ["ra", "dec", "mjd", "w1mpro", "w2mpro", "w1flux", "qual_frame"]
BOXSIZE = 25 # In Square Deg
BOXCENTER = (86.27208, -2.76164) # RA, Dec
QUALFRAMEMIN = 4

bxcntr = np.array(BOXCENTER)
side = np.sqrt(BOXSIZE)
halfside = side / 2

cmdstart = 'curl -v "'
url = "https://irsa.ipac.caltech.edu/TAP/async?"
query = f"QUERY=SELECT+{','.join(COLS)}+FROM+neowiser_p1bs_psd+WHERE+CONTAINS(POINT(ra,dec),BOX({bxcntr[0]},{bxcntr[1]},{side},{side}))=1+and+(qual_frame>4)+order+by+dec"
format_ = '&FORMAT=CSV'
end = '"'


cmd = cmdstart + url + query + format_ + end
reqaddress = url + query + format_
print(cmd)
# result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, text=True)
# print(result)
# locationurl = str(result.stdout)
# print(locationurl.split("Location: "))

# done = False
# while not done:
#     check = requests.get(locationurl + "/phase")
#     print(check)
#     time.sleep(10)
