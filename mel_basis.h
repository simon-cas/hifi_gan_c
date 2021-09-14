#include <stdio.h>
#include <math.h>

//float mel_filter_bank(float * mels_basis[513][80]);
void fft_frequencies(float * fftfreqs);
void obtain_mels(float * mels);
void mel_frequencies(float * mels, float * mel_f);

void mel_filter_bank(float mel_basis[513][80]){
    int i;
    int j;
    int sr = 22050;
    int n_fft = 1024;
    int n_mels = 80;
    int fmin = 0;
    int fmax = 8000;
    //float mel_basis[513][80];
    float fftfreqs[513];
    float mels[n_mels+2];
    float mel_f[n_mels+2];
    float fdiff[81];
    float ramps[513][82];
    float enorm[80];
    float lower;
    float upper;
    float temp;


    fft_frequencies(fftfreqs);
    obtain_mels(mels);
    mel_frequencies(mels, mel_f);

    for(i = 0; i < 81; i++){
        fdiff[i] = mel_f[i+1] - mel_f[i];
	//printf("%f,", fdiff[i]);
    }


    for(i = 0; i < 513; i++){
        for(j = 0; j < 82; j++){
	    ramps[i][j] = mel_f[j] - fftfreqs[i];
	}
    
    }


    for(i = 0; i < 80; i++){
	/* slaney norm */
	enorm[i] = 2.0/(mel_f[i+2] - mel_f[i]);
	for(j = 0; j < 513; j++){
            lower = -ramps[j][i]/fdiff[i];
	    upper = ramps[j][i+2] / fdiff[i+1];
	    //printf("(%f, %f), ", lower, upper);
	

            if (lower<=upper){
		temp = lower;
	    }
	    else{
	        temp = upper;
	    }

	    if(temp<0.0){
	        temp = 0.0;
	    }

	    //printf("%f,", temp);

	    mel_basis[j][i] = temp;
  	    mel_basis[j][i] *= enorm[i];
            //printf("%f,", weights[j][i]);
        }
    }


    //for(i = 0; i < 513; i++){
    //	printf("***********");
    //	for(j = 0; j < 80; j++){
    //        printf("%f,", mel_basis[i][j]);
    //	}
    //	printf("***********");
    //}
    //return mel_basis;
}


void fft_frequencies(float * fftfreqs){
    int i;
    for(i = 0; i < 512; i++){
        fftfreqs[i+1] = fftfreqs[i] + 11025.0/512.0;
    }
    //for(i = 0; i < 513; i++){
    //    printf("%f,", fftfreqs[i]);
    //}


}


void obtain_mels(float * mels){
    int i;
    for(i = 0; i < 81; i ++){
        mels[i+1] = mels[i] + 45.245640471924965/81.0;
    }
}

void mel_frequencies(float * mels, float * mel_f){
    int i;
    float f_min = 0.0;
    float f_sp = 200.0 / 3;
    
    float min_log_hz = 1000.0;
    float min_log_mel = min_log_hz / f_sp;
    float logstep = log(6.4) / 27.0;

    for(i = 0; i < 82; i++){
        mel_f[i] = f_sp * mels[i];
	if(mels[i] > min_log_mel){
	    mel_f[i] = min_log_hz * exp(logstep * (mels[i] - min_log_mel));
	}
    }

}



/* gcc mel_basis.c -lm */


