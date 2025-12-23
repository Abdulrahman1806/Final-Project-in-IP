
import cv2
import numpy as np



def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def overlay(img, text):
    cv2.putText(img, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2)

# Basic Filters


def blur_filter(img, k):
    return cv2.GaussianBlur(img, (k, k), 0)

def gray_filter(img):
    return to_gray(img)

def canny_filter(img):
    return cv2.Canny(to_gray(img), 100, 200)

def negative_filter(img):
    return 255 - img

def binary_threshold(img):
    gray = to_gray(img)
    _, th = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    return th

def sobel_filter(img):
    gray = to_gray(img)
    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    mag = cv2.magnitude(sx, sy)
    return np.uint8(mag / mag.max() * 255)

# Mean Filters

def arithmetic_mean(gray, k):
    return cv2.blur(gray, (k, k))

def geometric_mean(gray, k):
    gray = gray.astype(np.float32) + 1
    return np.exp(cv2.blur(np.log(gray), (k, k))).astype(np.uint8)

def harmonic_mean(gray, k):
    gray = gray.astype(np.float32) + 1
    return ((k*k) / cv2.blur(1/gray, (k, k))).astype(np.uint8)

def contraharmonic_mean(gray, k, Q=1.5):
    num = cv2.blur(gray**(Q+1), (k, k))
    den = cv2.blur(gray**Q, (k, k)) + 1e-9
    return (num / den).astype(np.uint8)

# Order Statistics Filters

def min_filter(gray, k):
    return cv2.erode(gray, np.ones((k, k), np.uint8))

def max_filter(gray, k):
    return cv2.dilate(gray, np.ones((k, k), np.uint8))

def midpoint_filter(gray, k):
    return ((min_filter(gray, k).astype(np.int16) +
             max_filter(gray, k).astype(np.int16)) // 2).astype(np.uint8)

def alpha_trimmed(gray, k, d=4):
    pad = k // 2
    padded = np.pad(gray, pad, mode='edge')
    out = np.zeros_like(gray)

    for i in range(gray.shape[0]):
        for j in range(gray.shape[1]):
            window = padded[i:i+k, j:j+k].flatten()
            window.sort()
            window = window[d//2: len(window)-d//2]
            out[i, j] = np.mean(window)
    return out.astype(np.uint8)

# Frequency Domain

def D_uv(rows, cols):
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    return np.sqrt((x - ccol)**2 + (y - crow)**2)

def DFT_process(gray, H):
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft = np.fft.fftshift(dft)
    dft[:, :, 0] *= H
    dft[:, :, 1] *= H
    img_back = cv2.idft(np.fft.ifftshift(dft))
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = img_back / img_back.max() * 255
    return img_back.astype(np.uint8)

#  Low / High Pass 
# Any point inside circle = 1  and out the circle is 0
def ILPF(gray, D0):
    D = D_uv(*gray.shape)
    return DFT_process(gray, (D <= D0).astype(np.float32))
# gaussian curve
def GLPF(gray, D0):
    D = D_uv(*gray.shape)
    return DFT_process(gray, np.exp(-(D**2)/(2*D0**2)))

def BLPF(gray, D0, n=2): # if  n increase it will be close to ideal and make balance between soft the noise and save the details
    D = D_uv(*gray.shape)
    return DFT_process(gray, 1/(1+(D/D0)**(2*n)))
# high pass it's a reverse of low pass
def IHPF(gray, D0): return 255 - ILPF(gray, D0)
def GHPF(gray, D0): return 255 - GLPF(gray, D0)
def BHPF(gray, D0): return 255 - BLPF(gray, D0)

# Band Reject / Pass 



# D0 = desired center of the band
# W = bandwidth
#result: A psychological image of the band's end , but the selected frequencies will disappears
def IBRF(gray, D0, W):
    D = D_uv(*gray.shape)
    H = np.ones(gray.shape)
    H[np.abs(D-D0) <= W/2] = 0
    return DFT_process(gray, H)
# same idea but allowing and blocking is seamless  , it reduces the selected frequencies
def GBRF(gray, D0, W):
    D = D_uv(*gray.shape) + 1e-9
    return DFT_process(gray, 1 - np.exp(-((D**2-D0**2)/(D*W))**2))
#   A flexible approach between ideal and gaussian 
# n = filter sharpness (higher - like ideal)
def BBRF(gray, D0, W, n=2):
    D = D_uv(*gray.shape) + 1e-9
    return DFT_process(gray, 1/(1+((D*W)/(D**2-D0**2))**(2*n)))
# reverse of reject it only requested band is allowed, and all other frequencies are blocked
def IBPF(gray, D0, W): return 255 - IBRF(gray, D0, W)
def GBPF(gray, D0, W): return 255 - GBRF(gray, D0, W)
def BBPF(gray, D0, W): return 255 - BBRF(gray, D0, W)

# Web_cam


cap = cv2.VideoCapture(0)
mode = None
k = 3
D0 = 30
W = 10

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = to_gray(frame)
    out = frame.copy()
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'): break
    elif key == ord('n'): mode = None
    elif key == ord('+'): k = min(k+2, 31)
    elif key == ord('-'): k = max(k-2, 1)
    elif key == ord(']'): D0 += 5
    elif key == ord('['): D0 = max(5, D0-5)
    elif key == ord('}'): W += 5
    elif key == ord('{'): W = max(5, W-5)

    elif key == ord('b'): mode='blur'
    elif key == ord('r'): mode='gray'
    elif key == ord('e'): mode='canny'
    elif key == ord('u'): mode='negative'
    elif key == ord('t'): mode='binary'
    elif key == ord('k'): mode='sobel'

    elif key == ord('a'): mode='arith'
    elif key == ord('g'): mode='geo'
    elif key == ord('h'): mode='harm'
    elif key == ord('m'): mode='contra'

    elif key == ord('z'): mode='median'
    elif key == ord('x'): mode='min'
    elif key == ord('y'): mode='max'
    elif key == ord('p'): mode='mid'
    elif key == ord('l'): mode='alpha'

    elif key == ord('1'): mode='ilpf'
    elif key == ord('2'): mode='glpf'
    elif key == ord('3'): mode='blpf'
    elif key == ord('4'): mode='ihpf'
    elif key == ord('5'): mode='ghpf'
    elif key == ord('6'): mode='bhpf'
    elif key == ord('7'): mode='ibrf'
    elif key == ord('8'): mode='gbrf'
    elif key == ord('9'): mode='bbrf'
    elif key == ord('o'): mode='ibpf'
    elif key == ord('i'): mode='gbpf'
    elif key == ord('j'): mode='bbpf'

    # Processing
    if mode == 'blur': out = blur_filter(frame, k)
    elif mode == 'gray': out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif mode == 'canny': out = cv2.cvtColor(canny_filter(frame), cv2.COLOR_GRAY2BGR)
    elif mode == 'negative': out = negative_filter(frame)
    elif mode == 'binary': out = cv2.cvtColor(binary_threshold(frame), cv2.COLOR_GRAY2BGR)
    elif mode == 'sobel': out = cv2.cvtColor(sobel_filter(frame), cv2.COLOR_GRAY2BGR)

    elif mode == 'arith': out = cv2.cvtColor(arithmetic_mean(gray, k), cv2.COLOR_GRAY2BGR)
    elif mode == 'geo': out = cv2.cvtColor(geometric_mean(gray, k), cv2.COLOR_GRAY2BGR)
    elif mode == 'harm': out = cv2.cvtColor(harmonic_mean(gray, k), cv2.COLOR_GRAY2BGR)
    elif mode == 'contra': out = cv2.cvtColor(contraharmonic_mean(gray, k), cv2.COLOR_GRAY2BGR)

    elif mode == 'median': out = cv2.cvtColor(cv2.medianBlur(gray, k), cv2.COLOR_GRAY2BGR)
    elif mode == 'min': out = cv2.cvtColor(min_filter(gray, k), cv2.COLOR_GRAY2BGR)
    elif mode == 'max': out = cv2.cvtColor(max_filter(gray, k), cv2.COLOR_GRAY2BGR)
    elif mode == 'mid': out = cv2.cvtColor(midpoint_filter(gray, k), cv2.COLOR_GRAY2BGR)
    elif mode == 'alpha': out = cv2.cvtColor(alpha_trimmed(gray, k), cv2.COLOR_GRAY2BGR)

    elif mode == 'ilpf': out = cv2.cvtColor(ILPF(gray, D0), cv2.COLOR_GRAY2BGR)
    elif mode == 'glpf': out = cv2.cvtColor(GLPF(gray, D0), cv2.COLOR_GRAY2BGR)
    elif mode == 'blpf': out = cv2.cvtColor(BLPF(gray, D0), cv2.COLOR_GRAY2BGR)
    elif mode == 'ihpf': out = cv2.cvtColor(IHPF(gray, D0), cv2.COLOR_GRAY2BGR)
    elif mode == 'ghpf': out = cv2.cvtColor(GHPF(gray, D0), cv2.COLOR_GRAY2BGR)
    elif mode == 'bhpf': out = cv2.cvtColor(BHPF(gray, D0), cv2.COLOR_GRAY2BGR)
    elif mode == 'ibrf': out = cv2.cvtColor(IBRF(gray, D0, W), cv2.COLOR_GRAY2BGR)
    elif mode == 'gbrf': out = cv2.cvtColor(GBRF(gray, D0, W), cv2.COLOR_GRAY2BGR)
    elif mode == 'bbrf': out = cv2.cvtColor(BBRF(gray, D0, W), cv2.COLOR_GRAY2BGR)
    elif mode == 'ibpf': out = cv2.cvtColor(IBPF(gray, D0, W), cv2.COLOR_GRAY2BGR)
    elif mode == 'gbpf': out = cv2.cvtColor(GBPF(gray, D0, W), cv2.COLOR_GRAY2BGR)
    elif mode == 'bbpf': out = cv2.cvtColor(BBPF(gray, D0, W), cv2.COLOR_GRAY2BGR)

    overlay(out, f"{mode} | k={k} | D0={D0} | W={W}")
    cv2.imshow("DSP LAB CAMERA", out)

cap.release()
cv2.destroyAllWindows()
