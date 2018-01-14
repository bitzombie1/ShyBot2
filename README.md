# ShyBot2
Ver 2 of Joel Hobbie's sculpture code 
This project includes code for facial detection and motor control. The facial detection utilizes OpenCV (ver 2) and runs on the UpBoard with a Ubuntu 14.04 system (Kernel 4.12.14 generic). Unlike version 1 of this project I opted to program with Python instead of C++. I am using multiple CV recognizers for front face, side face, and upper body in that order. I am using an Intel Realsense RS300 depth sensing camera so once I find a person, I can detect how far away that person is from the camera. This gives me the X,Y, and Z values of people in relation to the RS300 and then I send this information out of the serial connection to the Arduino Nano that controls multiple servo motors. This allows the sculpture to follow people in view with a pan/tilt controlled faux eyeball and retract its limbs if the person comes toward the sculpture thus giving it a shy behavior. 