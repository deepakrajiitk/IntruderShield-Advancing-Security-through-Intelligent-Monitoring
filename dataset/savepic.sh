#!/bin/bash

# Please note that this script will not kill ffmpeg program on its own. Please kill all ffmpeg programs manually, else they will continue to save pics

declare -a cameras=(
	[0]=ch24
	# [1]=ch25
	# [2]=ch26
	# [3]=ch27
	# [4]=ch28
	# [5]=ch29

# [0]=ch01
# [1]=ch05
)

# Delete the file where we will store PIDs of ffmpeg command, in case it exists
if [ -f ~/ffmpeg.pids ] 
then
	rm ~/ffmpeg.pids
fi

# Check presence of the folder structure which will be created, else create one
if [ ! -d "main" ] 
then
	mkdir main
else
	rm -rf main/
	mkdir main
fi

# Run for each camera

for cam in ${cameras[@]}
do
{

# Check if camera sub-directory exists

if [ ! -d "main/$cam" ]; then
        mkdir main/$cam
fi

# Run ffmpeg command in that directory

cd main/$cam

ffmpeg -hide_banner -y -rtsp_transport tcp -use_wallclock_as_timestamps 1 -i "rtsp://admin:Ntadg@7094@192.168.1.2:554/$cam/0"  -f segment -reset_timestamps 1 -r 1 -f image2 -strftime 1 $cam-%Y%m%dT%H%M%S.jpg 1> /dev/null 2>/dev/null &

# Save PID to kill it later
echo $! >> ~/ffmpeg.pids

#Go back to main directory from where command was run
cd ../../
}
done

