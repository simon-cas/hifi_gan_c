#include <stdio.h>
#include <time.h>
#include "kiss_fft/fft_compute.h"
#include "hifi_gan.h"
#include <stdlib.h>


int main(int argc, char **argv){
    if (argc!=3) {
        fprintf(stderr, "usage: %s <input speech> <output speech>\n", argv[0]);
    return 1;
    }
    int i, j, k;
    clock_t start, finish;
    double  duration;
    FILE *wavfile;
    long size;
    short win_size = 1024;
    short hop_size = 256;
    short frame_nums;
    short frame;
    short win_data[win_size];
    short hop_data[hop_size];
    float PI = 3.141592653589793;
    short data_in[1024];
    
    float * hann_window = (float*) calloc(win_size, sizeof(float));

    for(i = 0; i < win_size; i++){
        hann_window[i] = 0.5 * (1 - cos(2 * PI * i / win_size));
    }

    kiss_fft_cfg cfg = kiss_fft_alloc(win_size, 0, 0 ,0 );


    kiss_fft_cpx buf_in[win_size];
    kiss_fft_cpx buf_out[win_size];


    start = clock();

    wavfile = fopen(argv[1], "rb");

    fseek(wavfile, 0, SEEK_END);

    /*data size*/
    size = (ftell(wavfile) - 78)/2;
    
    /*skip wav header*/
    fseek(wavfile, 78, SEEK_SET);

    frame_nums = (size - win_size)/hop_size + 1;
    //printf("%d, %d, ", frame_nums, size);

    float pow_spec[frame_nums][513];
    float mel_spec[frame_nums][80];

    int first = 1;

    int frame_id = 0;

    while(1){
        if (first){
	    fread(win_data, sizeof(short), win_size, wavfile);

	    for(i = 0; i < win_size; i++){
                data_in[i] = win_data[i];
            }

	}
	else{
	    for(i = 0; i < 768; i++){
	        data_in[i] = data_in[i+256];
	    }
	    fread(hop_data, sizeof(short), hop_size, wavfile);
	    if (feof(wavfile)) break;

	    for(i = 0; i < 256; i++){
	        data_in[i+768] = hop_data[i];
	    }
	}

	for(i = 0; i < win_size; i++){
                buf_in[i].r = hann_window[i] * data_in[i] / 32768.0;
                buf_in[i].i = 0;
            }

	kiss_fft(cfg , buf_in , buf_out);

        for(i = 0; i < 513; i++){
            //printf("%f, ", buf_out[i].r);
        }


        for(i = 0; i < 513; i++){
	    pow_spec[frame_id][i] = sqrt(pow(buf_out[i].r, 2) + pow(buf_out[i].i, 2) + 1e-9);
	}


	first = 0;
	frame_id += 1;
    }
	fclose(wavfile);
    kiss_fft_free(cfg);

    for(i = 0; i < frame_nums; i++){
        for(j = 0; j < 80; j++){
	    for( k = 0; k < 513;k++){
	        mel_spec[i][j] += pow_spec[i][k] * mel_basis[j][k];
            }
	if(mel_spec[i][j] < 1e-5){
                mel_spec[i][j] = 1e-5;
            }
        mel_spec[i][j] = log(mel_spec[i][j]);
	}
    }

     for(i = 0; i < 80; i++){
         //printf("%f,", mel_spec[244][i]);
    }

    /*pre_convolution*/
    float * conv_pre_out = (float*) calloc(frame_nums * 128, sizeof(float));
    conv_pre(mel_spec, conv_pre_out, frame_nums);
    

    /*The first upsampling process*/
    float * ups0_out = (float*) calloc(frame_nums*8*64, sizeof(float));
    ups0(conv_pre_out, ups0_out, frame_nums); 

    float * resblock0_out = (float*) calloc(frame_nums*8*64, sizeof(float));
    resblock0(ups0_out, resblock0_out, frame_nums*8);

    float * resblock1_out = (float*) calloc(frame_nums*8*64, sizeof(float));
    resblock1(ups0_out, resblock1_out, frame_nums*8);

    float * resblock2_out = (float*) calloc(frame_nums*8*64, sizeof(float));
    resblock2(ups0_out, resblock2_out, frame_nums*8);

    for(i = 0; i < frame_nums * 8 *64; i++){
        resblock2_out[i] = (resblock0_out[i] + resblock1_out[i] + resblock2_out[i])/3;
    }

    //free(conv_pre_out);
    free(ups0_out);
    free(resblock0_out);
    free(resblock1_out);
    //conv_pre_out = NULL;
    ups0_out = NULL;
    resblock0_out = NULL;
    resblock1_out = NULL;



    /*The second upsampling process*/
    float * ups1_out = (float*) calloc(frame_nums*2048, sizeof(float));
    ups1(resblock2_out, ups1_out, frame_nums*8); //out frame nums become frame_nums*8 after the first upsampling process


    float * resblock3_out = (float*) calloc(frame_nums*2048, sizeof(float));
    resblock3(ups1_out, resblock3_out, frame_nums*64); //out frame nums become frame_nums*64 after the second upsampling process

    float * resblock4_out = (float*) calloc(frame_nums*2048, sizeof(float));
    resblock4(ups1_out, resblock4_out, frame_nums*64);

    float * resblock5_out = (float*) calloc(frame_nums*2048, sizeof(float));
    resblock5(ups1_out, resblock5_out, frame_nums*64);

    for(i = 0; i < frame_nums*2048; i++){
        resblock5_out[i] = (resblock3_out[i] + resblock4_out[i] + resblock5_out[i])/3;
    }

    free(resblock2_out);
    free(ups1_out);
    free(resblock3_out);
    free(resblock4_out);
    resblock2_out = NULL;
    ups1_out = NULL;
    resblock3_out = NULL;
    resblock4_out = NULL;


    /*The third upsampling process*/
    float * ups2_out = (float*) calloc(frame_nums*2048, sizeof(float));
    ups2(resblock5_out, ups2_out, frame_nums*64); //out frame nums become frame_nums*64 after the sencond upsampling process

    
    float * resblock6_out = (float*) calloc(frame_nums*2048, sizeof(float));
    resblock6(ups2_out, resblock6_out, frame_nums*128); //out frame nums become frame_nums*128 after the third upsampling process

    float * resblock7_out = (float*) calloc(frame_nums*2048, sizeof(float));
    resblock7(ups2_out, resblock7_out, frame_nums*128);

    float * resblock8_out = (float*) calloc(frame_nums*2048, sizeof(float));
    resblock8(ups2_out, resblock8_out, frame_nums*128);

    for(i = 0; i < frame_nums*2048; i++){
        resblock8_out[i] = (resblock6_out[i] + resblock7_out[i] + resblock8_out[i])/3;
    }


    free(resblock5_out);
    free(ups2_out);
    free(resblock6_out);
    free(resblock7_out);
    resblock5_out = NULL;
    ups2_out = NULL;
    resblock6_out = NULL;
    resblock7_out = NULL;

    
    /*The fourth upsampling process*/
    float * ups3_out = (float*) calloc(frame_nums*2048, sizeof(float));
    ups3(resblock8_out, ups3_out, frame_nums*128); //in frame nums become frame_nums*64 after the sencond upsampling process


    float * resblock9_out = (float*) calloc(frame_nums*2048, sizeof(float));
    resblock9(ups3_out, resblock9_out, frame_nums*256); //out frame nums become frame_nums*256 after the third upsampling process

  
    float * resblock10_out = (float*) calloc(frame_nums*2048, sizeof(float));
    resblock10(ups3_out, resblock10_out, frame_nums*256);

    float * resblock11_out = (float*) calloc(frame_nums*2048, sizeof(float));
    resblock11(ups3_out, resblock11_out, frame_nums*256);


    for(i = 0; i < frame_nums*2048; i++){
        resblock11_out[i] = (resblock9_out[i] + resblock10_out[i] + resblock11_out[i])/3;
    }

    free(resblock8_out);
    free(ups3_out);
    free(resblock9_out);
    free(resblock10_out);
    resblock8_out = NULL;
    ups3_out = NULL;
    resblock9_out = NULL;
    resblock10_out = NULL;

    float * conv_post_out = (float*) calloc(frame_nums*256, sizeof(float));
    conv_post(resblock11_out, conv_post_out, frame_nums*256);


    short wav_out[frame_nums*256];

    for(i = 0; i < frame_nums*256; i++){
        wav_out[i] = (short) (tanh(conv_post_out[i]) * 32768.0);
    }


    FILE * fout;
    fout = fopen(argv[2], "wb");
    fwrite(wav_out, sizeof(short), frame_nums*256, fout);
    fclose(fout);
    free(resblock11_out);
    resblock11_out = NULL;
    free(conv_post_out);
    conv_post_out = NULL;
    
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("%f seconds\n", duration);

    return 0;
}



