//-----------------------------------------
// Basit-Seri Yapay Sinir Agi Gerceklemesi
// Huseyin Mutlu
//-----------------------------------------
 
#include "stdafx.h"
#include <iostream>
#include <vector>
#include <time.h>
 
 
#define INPUT_COUNT 2
#define HIDDEN_COUNT 4
#define OUTPUT_COUNT 1
#define MAX_EPOCH 500
#define RESULT_PER_EPOCH 100
#define LEARNING_RATE 0.9
#define TRAIN_SIZE 100
#define ALPHA 0.9
#define TEST_OPERATOR ^
#define RANDOM_DOUBLE randomDouble()
#define LINE "\n---------------------------------------\n"
 
using namespace std;
 
 
// 2-layer neural network
typedef struct NN2 {
        unsigned int inputCount;
        unsigned int hiddenCount;
        unsigned int outputCount;
       
        double* input;
        double* hidden;
        double* output;
 
        double** weight_i2h; // input to hidden layer weights
        double** weight_h2o; // hidden to output weights;
 
        double (*activator)(double);
        double (*delta)(double, double);
} NN2;
 
void setupNN2(NN2* nn2) ;
void printWeights(NN2* nn2);
double activator(double);
double randomDouble();
void randomizeWeights(NN2* nn2);
void passForward(NN2*, unsigned char*);
double delta(double, double);
void train(NN2*, unsigned char*);
void printNN2(NN2*);
 
int roundDouble(double number){return (number >= 0) ? (int)(number + 0.5) : (int)(number - 0.5);}
 
 
int _tmain(int argc, _TCHAR* argv[]) {
        srand(time(NULL)); //rastgeleliği artırmak için
    // EĞİTİM SETİNİ OLUŞTUR
    // Giriş dizisi
    unsigned char inputArray[TRAIN_SIZE][INPUT_COUNT + 1];
    // Sonuç dizisi
    unsigned char outputArray[TRAIN_SIZE];
 
    for(int t = 0; t < TRAIN_SIZE; t++) {
        inputArray[t][0] = rand() % 2; // Rastgele olarak 1 veya 0 seç
        inputArray[t][1] = rand() % 2; // Rastgele olarak 1 veya 0 seç
        inputArray[t][2] = 1; // bias
 
    //Sonuç dizisini oluştur
        outputArray[t] = inputArray[t][0] TEST_OPERATOR inputArray[t][1];
    }
    //Sinir ağu kuruluyor:
 
    //Ağı kur
    NN2 nn2;
    setupNN2(&nn2);
 
    //Iterasyon dizileri
    unsigned char inputSet[INPUT_COUNT + 1];
    unsigned char outputSet[OUTPUT_COUNT];
 
    cout << "OK\n";
    cout << LINE << "EXECUTION PHASE" << LINE;
    cout << "Training started [" << MAX_EPOCH << " epoch]...";
    clock_t start = clock(); // başlangıç saat vuruşu
 
    // MAX_EPOCH: bütün eğitim setinin genel tekrar sayısı
    for(int e=0; e < MAX_EPOCH; e++) {
        //Eğitim itersyonları
        for(int t=0; t < TRAIN_SIZE; t++) {
 
            //kullanılacak giriş verileri
            for(int i=0; i < nn2.inputCount; i++) {
                inputSet[i] = inputArray[t][i];
            }
 
            //beklenen çıkış verileri
            for(int o=0; o < nn2.outputCount; o++) {
                outputSet[o] = outputArray[t];
            }
 
            //ileri git
            passForward(&nn2,inputSet);
 
            //eğit
            train(&nn2, outputSet);
 
        }
 
    }
    //eğitim süresi
    double execTime = ((double)clock() - start)/CLOCKS_PER_SEC;
    printf("OK (%3.2f secs)", execTime);
 
    getchar();
    return 0;
 
       
}
 
void setupNN2(NN2* nn2) {
 
        nn2->inputCount = INPUT_COUNT + 1;
        nn2->hiddenCount = HIDDEN_COUNT + 1;
        nn2->outputCount = OUTPUT_COUNT;
 
        //Bellek atamalari
        nn2->input = (double*)malloc(nn2->inputCount * sizeof(double));
        nn2->hidden = (double*)malloc(nn2->hiddenCount * sizeof(double));
        nn2->output = (double*)malloc(nn2->outputCount * sizeof(double));
         
        nn2->weight_i2h = (double**)malloc(nn2->inputCount * sizeof(double*));
        for(int i=0; i<nn2->inputCount; i++) {
                nn2->weight_i2h[i] = (double*)calloc(2 * nn2->hiddenCount, sizeof(double));
        }
         
        nn2->weight_h2o = (double**)malloc(nn2->hiddenCount * sizeof(double*));
        for(int h=0; h<nn2->hiddenCount; h++) {
                nn2->weight_h2o[h] = (double*)calloc(2 * nn2->outputCount, sizeof(double));
        }
               
        //Set activation function
        nn2->activator = &activator;
        nn2->delta = &delta;
       
        //Initialize the weights      
        randomizeWeights(nn2);
 
}
 
/**
* Ağırlık dizilerini ekranda göstermek için
*/
void printWeights(NN2* nn2) {
 
        for(int i=0; i < nn2->inputCount; i++) {
                for(int h = 0; h < nn2->hiddenCount; h++) {
                        printf("\n[%di, %dh] = %f", i,h, nn2->weight_i2h[i][h]);
                }
                printf("\n---");
        }
       
        for(int h = 0; h < nn2->hiddenCount; h++) {
                for(int o = 0; o < nn2->outputCount; o++) {
                        printf("\n[%dh, %do] = %f", h,o, nn2->weight_h2o[h][o]);
                }
                printf("\n---");
        }
               
       
}
 
/**
* Ağdaki düğümleri göstermek için
*/
void printNN2(NN2* nn2) {
 
        cout << LINE;
        cout << "\nINPUT:  ";
        for(int i = 0; i < nn2->inputCount; i++) {
                printf("%f\t",nn2->input[i]);
        }
        cout << "\nHIDDEN: ";
        for(int i = 0; i < nn2->hiddenCount; i++) {
                printf("%f\t",nn2->hidden[i]);
        }
        cout << "\nOUTPUT: ";
        for(int i = 0; i < nn2->outputCount; i++) {
                printf("%f\t",nn2->output[i]);
        }
        cout << "\n";
}
 
// Delta: [d/dx(1/1+e^-x)]*errorSum
double delta(double value, double errorSum) {
        return value * (1 - value) * errorSum;
}
 
// İleri besle
void passForward(NN2* nn2, unsigned char* inputArray) {
 
        for(int i=0; i < nn2->inputCount; i++) {
                nn2->input[i] = inputArray[i]; // giriş düğümlerine verileri ata
        }
 
        //her bir gizli düğüme düşen ağırlıklı toplam
        double sum;
 
        for(int h=0; h < nn2->hiddenCount; h++) {
                sum = 0;
                for(int i=0; i < nn2->inputCount; i++) {
                        sum += nn2->weight_i2h[i][h] * nn2->input[i];
                }
                // Ağırlıklı toplam aktivasyon fonksiyonundan geçerek düğüme atanıyor.
                nn2->hidden[h] = nn2->activator(sum);
        }
 
        //Aynı işlemler gizli düğümler ile çıkış düğümleri arasında uygulanıyor.
        for(int o=0; o < nn2->outputCount; o++) {
                sum=0;
                for(int h=0; h < nn2->hiddenCount; h++) {
                        sum += nn2->hidden[h] * nn2->weight_h2o[h][o];
                }
                nn2->output[o] = nn2->activator(sum);
        }
 
}
 
void train(NN2* nn2, unsigned char* targetArray) {
        //Çıkış ve gizli düğümlere ait delta değerlerini tutacak diziler
        double* deltaOutputs = (double*)malloc(nn2->outputCount * sizeof(double));
        double* deltaHiddens = (double*)malloc(nn2->hiddenCount * sizeof(double));
         
        //Delta degerlerinin hesaplanması
        double errorSum; //Hata toplamı
        for(int o=0; o < nn2->outputCount; o++) {
                // hata = beklenen değer - çıkış değeri
                errorSum = targetArray[o] - nn2->output[o];
                // çıkışın delta değeri
                deltaOutputs[o] = nn2->delta(nn2->output[o], errorSum);
        }
         
        for(int h=0; h < nn2->hiddenCount; h++) {
                errorSum = 0.0;
                for(int o=0; o < nn2->outputCount; o++) {
                        //Çıkış delta değeri ile aradaki ağırlık çarpılarak toplanıyor
                        errorSum += deltaOutputs[o] * nn2->weight_h2o[h][o];
                }
                //Gizli düğümün delta değeri
                deltaHiddens[h] = nn2->delta(nn2->hidden[h], errorSum);
        }
        //Deltalar hesaplandıktan sonra ağırlıkları ayarlama aşamasına geçiyoruz.
         
        double *lastDelta;
        double newDelta;
         
        // Çıkış ve gizli düğümler arası ağırlıkların belirlenmesi
        for(int o=0; o < nn2->outputCount; o++) {
                for(int h=0; h < nn2->hiddenCount; h++) {
                        lastDelta = &nn2->weight_h2o[h][nn2->outputCount + o];
                        // Yerel minimumlardan kaçınabilmek için momentum ekleniyor.
                        newDelta = *lastDelta * ALPHA + (deltaOutputs[o] * nn2->hidden[h])*(1 - ALPHA);
                        nn2->weight_h2o[h][o] += LEARNING_RATE * newDelta;
                        *lastDelta = newDelta;
                }
                }
 
                // Gizli düğümler ile giriş düğümleri arasındaki ağırlıklar
                for(int i=0; i < nn2->inputCount; i++) {
                        for(int h=0; h < nn2->hiddenCount; h++) {
                                //Set the weight of i2h
                                lastDelta = &nn2->weight_i2h[i][nn2->hiddenCount + h];
                                newDelta = *lastDelta * ALPHA + (deltaHiddens[h] * nn2->input[i])*(1 - ALPHA);
                                nn2->weight_i2h[i][h] += LEARNING_RATE * newDelta;
                                *lastDelta = newDelta;
 
                        }
                }
}
 
void randomizeWeights(NN2* nn2) {
 
        for(int i = 0; i < nn2->inputCount; i++) {
                for(int h = 0; h < nn2->hiddenCount; h++) {
                        nn2->weight_i2h[i][h] = (RANDOM_DOUBLE * 4) - 2;
                }
        }
 
        for(int h = 0; h < nn2->hiddenCount; h++) {
                for(int o = 0; o < nn2->outputCount; o++) {
                        nn2->weight_h2o[h][o] = (RANDOM_DOUBLE * 4) - 2;
                }
        }
 
}
 
 
double activator(double input) {
       
        return 1.0 / (1.0 + exp(-input));
 
}
 
double randomDouble() {
        return (double)rand()/RAND_MAX;
}
