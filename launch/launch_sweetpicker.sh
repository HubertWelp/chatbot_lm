#!/bin/bash

# Default to original system if no argument is provided
USE_YOLO=${1:-0}

# Display which object detector will be used
DETECTOR_TYPE=$([ "$USE_YOLO" -eq 1 ] && echo 'YOLO' || echo 'Original')
DETECTOR_CMD=$([ "$USE_YOLO" -eq 1 ] && echo "source /home/student/git/SP4/SweetPicker4/SP4Objekterkenner/venv_SP4Objekterkenner/bin/activate && python3 /home/student/git/SP4/SweetPicker4/SP4Objekterkenner/SP4Objekterkenner.py" || echo "python3.9 /home/student/git/SP4/SweetPicker4/SP4Objekterkenner/SP3Objekterkenner.py /home/student/git/SP4/SweetPicker4/SP4Bildanalysator/SP3Bilderkennung/aktuelleSzene.jpg")

# Launch all components in tabs
gnome-terminal \
    --tab --title="ROS Core" \
        --command="bash -c 'echo -e \"\e[1;34m===== ROS Core =====\e[0m\"; source ~/catkin_ws/devel/setup.bash; roscore; exec bash'" \
    --tab --title="Chatbot Orchestrator" \
        --command="bash -c 'echo -e \"\e[1;34m===== Chatbot Orchestrator =====\e[0m\"; sleep 3; source ~/catkin_ws/devel/setup.bash; source ~/catkin_ws/src/chatbot_lm/.venv/bin/activate; cd ~/catkin_ws/src/chatbot_lm/src; python chatbot_orchestrator.py; exec bash'" \
    --tab --title="ROS Bridge" \
        --command="bash -c 'echo -e \"\e[1;34m===== ROS Bridge =====\e[0m\"; sleep 3; source ~/catkin_ws/devel/setup.bash; rosrun chatbot_lm ros_bridge.py; exec bash'" \
    --tab --title="SP4 Koordinator" \
        --command="bash -c 'echo -e \"\e[1;34m===== SP4 Koordinator =====\e[0m\"; sleep 4; source ~/catkin_ws/devel/setup.bash; rosrun SP4Koordinator SP4Koordinator; exec bash'" \
    --tab --title="SP4 Bildanalysator" \
        --command="bash -c 'echo -e \"\e[1;34m===== SP4 Bildanalysator =====\e[0m\"; sleep 5; /home/student/git/SP4/SweetPicker4/build-SP4Bildanalysator-Desktop_Qt_6_2_3_GCC_64bit-Debug/SP4Bildanalysator; exec bash'" \
    --tab --title="SP4 Admin" \
        --command="bash -c 'echo -e \"\e[1;34m===== SP4 Admin =====\e[0m\"; sleep 5; echo \"Passwort steht in konfig.ini (Wahrscheinlich SP4)\"; /home/student/git/SP4/SweetPicker4/build-SP4Admin-Desktop_Qt_6_2_3_GCC_64bit-Debug/SP4Admin; exec bash'" \
    --tab --title="SP4 Objekterkenner ($DETECTOR_TYPE)" \
        --command="bash -c 'echo -e \"\e[1;34m===== SP4 Objekterkenner ($DETECTOR_TYPE) =====\e[0m\"; sleep 6; $DETECTOR_CMD; exec bash'"

echo -e "\e[1;32mSweetPicker system launched with $DETECTOR_TYPE object detector and Chatbot integration.\e[0m"
