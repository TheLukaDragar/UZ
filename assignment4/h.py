#import lib for AES
from Crypto.Cipher import AES 






iv ="IV9zVgsPSwuVbbYjgs5hGTrb7lG953aO"

message="FMpxqPYXvtbPsdGfO2JWXv28mPiS7jRHzoSocTFfkLvZG76EaB6E761qD4OWwrvo"



#extract the key by using the iv ECB mode
key = AES.new(iv, AES.MODE_ECB).decrypt(iv)



