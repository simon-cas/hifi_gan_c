# hifi_gan_c
A purely header only c version of hifi-gan

# how to run
1. sh compile.sh
2.  ./hifi_gan_demo input.wav output.raw (for example, ./hifi_gan_demo p234_001_22k.wav output.raw)
3. sox -t raw -c 1 -e signed-integer -b 16 -r 22050 output.raw output.wav

# things to do
Computing acceleration
