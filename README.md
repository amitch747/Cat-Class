<H1 align="center">
CatClass
</H1>
<p align="center">
  <video src="./fullbod.mp4" width="100%" autoplay loop muted playsinline></video>
</p>


# UNDER CONSTRUCTION


## Troubleshooting
Setting up a camera in WSL2 is tricky
In powershell: 
```
usbipd list
usbipd attach --wsl --busid {BUSID}
```
In WSL:
```
lsusb
ls /dev/video*
sudo chmod 666 /dev/video0
```


## Jetson Setup
Jetson should already have required libraries so just run

`python3 -m scripts.convert_to_trt`

`python3 trt_inference.py`
