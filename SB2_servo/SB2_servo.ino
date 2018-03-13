/*
 * Shybot ver 2 servo control code for Arduino. Controls six 12v linear servos. Three servos are tied to 
 * pin 6 control line. Command input is expected on USB serial in the format "g<x-axis/y-axis/z-axis". Example:
 * "g150/70/10422" . z-axis values range from around 1500 to 22000 but we clamp values over 16000 and values
 * under 7000. 
 */
#include <Servo.h>


#define BUFFSIZE 24

#define UPBOARDONPIN A7       //reads 3.3v on UP-board to see if booted up
#define LEDSWPIN 2            // switches mosfet control for LEDs
// servo drive pins **************
#define SERVOX_PIN 11
#define SERVOY_PIN 10
#define SERVOZIRIS_PIN 9
#define SERVOZ_PIN 6
// servo home/limits
#define X_L 60
#define X_H 135
#define Y_L 47
#define Y_H 130
#define Z_IRIS_L 50
#define Z_IRIS_H 125
#define Z_L 60
#define Z_H 130

char inString[BUFFSIZE];          // Serial command buffer
bool serialEcho = false;          // do, or do not, repeat after me 

Servo servo_x, servo_y, servo_zIris, servo_z ; 

int curr_x, curr_y, curr_z; 
 
void setup() {

  Serial.begin(9600);  //connect to serial
  // servo connect setup ***********************
  servo_x.attach(SERVOX_PIN);
  servo_y.attach(SERVOY_PIN);
  servo_zIris.attach(SERVOZIRIS_PIN);
  servo_z.attach(SERVOZ_PIN);
  
  pinMode(LEDSWPIN,OUTPUT);
  pinMode(13,OUTPUT);
  
  delay(10);
  digitalWrite(LEDSWPIN, HIGH);
  Serial.println("ok");
}

void loop() {
  if(serialCheck()){ commandParser(inString);}
  
  if(analogRead(UPBOARDONPIN) > 400){
    digitalWrite(LEDSWPIN, HIGH);}
    else{digitalWrite(LEDSWPIN, LOW);}
  
  if(millis() % 60000 == 0){Serial.println("ok");} //

//  encChk();
 
}

// servo action handlers ************************************************************
void moveSculpt(){
  /*
  Serial.print("curr_x: "); Serial.println(curr_x);
  Serial.print("curr_y: "); Serial.println(curr_y);
  Serial.print("curr_z: "); Serial.println(curr_z);
  */
  int z_iris = map(curr_z, 7000, 16000, Z_IRIS_L, Z_IRIS_H);
  int z = map(curr_z, 16000, 7000, Z_L, Z_H);
  servo_zIris.write(z_iris); servo_z.write(z);
  delay(10);
  servo_x.write(map(curr_x,320,0,X_L,X_H));
  delay(10);
  servo_y.write(map(curr_y,0,240,Y_L,Y_H));
  delay(10);
  
}

// our regular serial functions *****************************
bool serialCheck() {
  int i=0;
  if (Serial.available()) {
    delay(30);
    while(Serial.available()) {
      inString[i++] = Serial.read();
    }
    inString[i++]='\0';  
    
    if (serialEcho) {
      Serial.println(inString);
    }
    return true;
  }
  return false;
}

void commandParser(char * command){
  int i =0;
  if(command[i] == 'h'){delay(1);} // home command
  else if(command[i] == 'q'){delay(1);}
  else if(command[i] == 's'){delay(1);}
  else if(command[i] == 'g'){
    char one_str[11]={}; int x =0; long one_l = 0;
    char two_str[11]={}; int y =0; long two_l =0;
    char three_str[11]={};  int z =0; long three_l =0;
    while(command[i] != '/' && i<BUFFSIZE) {
      i++;
      one_str[x] = command[i];
      x++;
    }
    one_str[x+1] = '\0'; 
    one_l = atol(one_str);
       
    i++;  // skip the forward slash
    
    while(command[i] != '/' && i<BUFFSIZE) {    
      two_str[y] = command[i];
      i++;
      y++;
    }
    two_str[y+1] = '\0'; 
    two_l = atol(two_str);
    
    i++;  // skip the forward slash

    while(command[i] != '/' && i<BUFFSIZE) {    
      three_str[z] = command[i];
      i++;
      z++;
    }
    three_str[z+1] = '\0'; 
    three_l = atol(three_str);

    curr_x = int(one_l); //map(int(one_l),640,0,400,600);
    curr_y = int(two_l); //map(int(three_l),0,480,600,400);

    int inz = int(three_l);
    if(inz > 16000){ inz = 16000;}
    else if(inz <7000){inz = 7000;}
    curr_z = inz; 
    
    
    moveSculpt();
    Serial.println("ok"); // send ack command
    
  }
} // end commandParser()
