int counter = 0;
int upCounter=LOW;
int setInput = LOW;
int chooseNum = LOW;
int inputCounter = 0;
int switchTrain =LOW;
int binary[5][8] = {{1,0,0,1,1,1,1,1},{0,1,0,0,0,0,0,1},{1,1,0,0,0,0,1,0},{0,1,0,1,0,0,1,1},{1,0,1,0,0,1,0,0}};
int rand1;
const int firstLayer = 4;
const int secondLayer = 2;
float weights2[secondLayer][firstLayer];
float weights1[firstLayer][8];
float weights2Transpose[firstLayer][secondLayer];
float biases1[firstLayer];
float biases2[secondLayer];
float activations2[secondLayer];
float activations1[firstLayer];
float zs2[secondLayer];
float zs1[firstLayer];
int x[6][8] = {{1,0,0,1,1,1,1,1},{0,1,0,0,0,0,0,1},{1,1,0,0,1,0,1,1},{1,0,1,0,0,1,0,1},{1,0,1,0,1,0,1,0},{0,0,1,1,1,0,1,1}};
int y[6][2] = {{0,1},{0,1},{0,1},{1,0},{1,0},{0,1}};
float u = 0.8;
int epochs = 10;    
float dot = 0;
float z;
float delta[secondLayer];
float nebla[firstLayer];
float delta_avg[8];
float nebla_avg[8];
float input_avg[8];
float input[8];
void setup() {
  Serial.begin(9600);
  for(int i =2;i<=9;i++){
    pinMode(i,OUTPUT);
    digitalWrite(i,LOW);
    }
    pinMode(10,INPUT);
    pinMode(13,INPUT);
    pinMode(12,INPUT);
    pinMode(1,INPUT);
    pinMode(11,OUTPUT);
 for(int i=0;i<secondLayer;i++){
  for(int j=0;j<firstLayer;j++){
     rand1 = random(-35,35);  
     weights2[i][j] = (float)rand1/100;
    }
    biases2[i] = (float)rand1/100;
  }   
 for(int i=0;i<firstLayer;i++){
  for(int j=0;j<8;j++){
     rand1 = random(-45,45);  
     weights1[i][j] = (float)rand1/100;
    }
    biases1[i] = (float)rand1/100;
  }
}

void loop() {
  digitalWrite(11,LOW);
  upCounter = digitalRead(13);
  if(upCounter==HIGH) upCounterFunc();
  setInput = digitalRead(12);
  if(setInput == HIGH){
    for(int j =0;j<8;j++){
      input[j] = digitalRead(j+2);
      }      
    feedForward(0,0);            
    for(int i=2;i<=9;i++){
    digitalWrite(i,myRound(activations2[i-secondLayer]));
    delay(10);
    }
              }
  switchTrain = digitalRead(10);
  if(switchTrain == HIGH){
    for(int i=2;i<=9;i++){
    digitalWrite(i,input[i-2]);
    delay(10);
    }
    gradientDescent();
    }
    
}
void gradientDescent(){
  digitalWrite(11,HIGH);
  delay(15);
  for(int i=0;i<epochs;i++){
    for(int k =0;k<6;k++){
      feedForward(1,k);
      }
    }
    switchTrain = LOW;
    delay(150);
    digitalWrite(11,LOW);
  }
void feedForward(int train,int index){
    if(train == 1){      
      dot = 0;
      for(int i =0;i<firstLayer;i++){
        for(int j=0;j<8;j++){
          dot= dot + weights1[i][j] * x[index][j];
          }
          z = dot + biases1[i];
          zs1[i] = z;
          activations1[i] = sigmoid(z);
          dot = 0;
        }
    
      dot = 0;
      for(int i =0;i<secondLayer;i++){
        for(int j=0;j<firstLayer;j++){
          dot= dot + weights2[i][j] * activations1[j];
          }
          z = dot + biases2[i];
          zs2[i] = z;
          activations2[i] = sigmoid(z);
          dot = 0;
         }
      for(int i = 0;i<secondLayer;i++){
        delta[i] = activations2[i] - y[index][i];    
        }
    dot = 0;    
    transpose();    
    for(int i = 0;i<firstLayer;i++){
      for(int j=0;j<secondLayer;j++){
        dot = dot + weights2Transpose[i][j] * delta[j]; 
        }
        nebla[i] = dot * sigmoidPrime(zs1[i]);
        dot = 0;
      }
      backProp(index);
      }
      else{
        dot = 0;
        for(int i =0;i<firstLayer;i++){
          for(int j=0;j<8;j++){
          dot= dot + weights1[i][j] * input[j];
          };
        z = dot + biases1[i];
        activations1[i] = sigmoid(z);
        dot = 0;
    }
    
    dot = 0;
      for(int i =0;i<secondLayer;i++){
          for(int j=0;j<firstLayer;j++){
            dot= dot + weights2[i][j] * activations1[j];
            }
          z = dot + biases2[i];
          activations2[i] = sigmoid(z);
          dot = 0;
          Serial.println(activations2[i]);    }      
        }      
  }

void backProp(int index){
   float d_sum=0.0;
   float in_sum = 0.0;
   float n_sum=0.0;
   /*for(int i = 0;i<8;i++){
    for(int j = 0;j<6;j++){
      d_sum = d_sum+delta[j][i];
      in_sum = in_sum+x[j][i];
      }
      delta_avg[i] = d_sum/6.0;
      input_avg[i] = in_sum/6.0;
    }   
   for(int i = 0;i<5;i++){
    for(int j = 0;j<6;j++){
      n_sum = n_sum+nebla[j][i];
      }
      nebla_avg[i] = n_sum/6.0;
    } */
   for(int i=0;i<secondLayer;i++){
          for(int j = 0;j<firstLayer;j++){
            weights2[i][j] = weights2[i][j] - (u * delta[i] * activations1[j]);            
            }
        biases2[i] = delta[i];
   }

    
   for(int i=0;i<firstLayer;i++){
    for(int j = 0;j<8;j++){
      weights1[i][j] = weights1[i][j] - (u*nebla[i] * x[index][j]);      
      }
      biases1[i] = nebla[i];
    }
    }
int myRound(float num){
  if(num>=0.5) return 1;
  else return 0;
  }

float sigmoid(float z){
  return 1/(1+exp(-1*z));
  }
float sigmoidPrime(float z){
    return sigmoid(z)*(1-sigmoid(z));
  }

void upCounterFunc(){    
    switch(counter){
     case 5:
              stateUpdate(binary[0][0],binary[0][1],binary[0][2],binary[0][3],binary[0][4],binary[0][5],binary[0][6],binary[0][7]);
        counter = 0;
        break;
     default:
      stateUpdate(binary[counter+1][0],binary[counter+1][1],binary[counter+1][2],binary[counter+1][3],binary[counter+1][4],binary[counter+1][5],binary[counter+1][6],binary[counter+1][7]);
      counter+=1;
      break;      
    }
    }
void transpose(){
  float temp[40];
  int index = 0;
  for(int i =0;i<secondLayer;i++){
    for(int j=0;j<firstLayer;j++){
      temp[index] = weights2[i][j];
      index++;
      }
    }
  index = 0;
  for(int i =0;i<firstLayer;i++){
    for(int j=0;j<secondLayer;j++){
      weights2Transpose[i][j] = temp[index];
      index++;
      }
    }
  }
 void stateUpdate(int f,int s, int t,int fou,int fif,int six,int sev,int ei){
            digitalWrite(2,f);
            digitalWrite(3,s);
            digitalWrite(4,t);
            digitalWrite(5,fou);
            digitalWrite(6,fif);
            digitalWrite(7,six);
            digitalWrite(8,sev);
            digitalWrite(9,ei);
            delay(250);           
  }
