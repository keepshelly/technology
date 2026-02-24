#include <stdio.h>
#include <math.h>
#include <complex.h>
#include "cblass.h"
#define EPS 1e-6
int ok = 1;
#define CHECK(name,cond) do { if (cond) printf("%-8s -> [ OK ]\n", name); else { printf("%-8s : [ FAILED ]\n", name); ok = 0; } } while(0)
void test_gemm(){int n=2;float sa=1,sb=0;double da=1,db=0;float ca[2]={1,0},cb[2]={0,0};double za[2]={1,0},zb[2]={0,0};
float sA[4]={1,0,0,1},sB[4]={2,0,0,2},sC[4]={0};double dA[4]={1,0,0,1},dB[4]={2,0,0,2},dC[4]={0};
float complex cA[4]={1,0,0,1},cB[4]={2,0,0,2},cC[4]={0};double complex zA[4]={1,0,0,1},zB[4]={2,0,0,2},zC[4]={0};
cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n,n,n,sa,sA,n,sB,n,sb,sC,n);CHECK("sgemm",fabsf(sC[0]-2)<1e-4);
cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n,n,n,da,dA,n,dB,n,db,dC,n);CHECK("dgemm",fabs(dC[0]-2)<EPS);
cblas_cgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n,n,n,ca,cA,n,cB,n,cb,cC,n);CHECK("cgemm",cabsf(cC[0]-2)<1e-4);
cblas_zgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,n,n,n,za,zA,n,zB,n,zb,zC,n);CHECK("zgemm",cabs(zC[0]-2)<EPS);}
void test_symm_hemm(){int n=2;float sa=1,sb=0;double da=1,db=0;float ca[2]={1,0},cb[2]={0,0};double za[2]={1,0},zb[2]={0,0};
float sA[4]={1,0,0,1},sB[4]={2,0,0,2},sC[4]={0};double dA[4]={1,0,0,1},dB[4]={2,0,0,2},dC[4]={0};
float complex cA[4]={1,0,0,1},cB[4]={2,0,0,2},cC[4]={0};double complex zA[4]={1,0,0,1},zB[4]={2,0,0,2},zC[4]={0};
cblas_ssymm(CblasRowMajor,CblasLeft,CblasUpper,n,n,sa,sA,n,sB,n,sb,sC,n);CHECK("ssymm",fabsf(sC[0]-2)<1e-4);
cblas_dsymm(CblasRowMajor,CblasLeft,CblasUpper,n,n,da,dA,n,dB,n,db,dC,n);CHECK("dsymm",fabs(dC[0]-2)<EPS);
cblas_csymm(CblasRowMajor,CblasLeft,CblasUpper,n,n,ca,cA,n,cB,n,cb,cC,n);CHECK("csymm",cabsf(cC[0]-2)<1e-4);
cblas_zsymm(CblasRowMajor,CblasLeft,CblasUpper,n,n,za,zA,n,zB,n,zb,zC,n);CHECK("zsymm",cabs(zC[0]-2)<EPS);
cblas_chemm(CblasRowMajor,CblasLeft,CblasUpper,n,n,ca,cA,n,cB,n,cb,cC,n);CHECK("chemm",cabsf(cC[0]-2)<1e-4);
cblas_zhemm(CblasRowMajor,CblasLeft,CblasUpper,n,n,za,zA,n,zB,n,zb,zC,n);CHECK("zhemm",cabs(zC[0]-2)<EPS);}
void test_syrk_herk(){int n=2;float sa=1,sb=0;double da=1,db=0;float ca[2]={1,0},cb[2]={0,0};double za[2]={1,0},zb[2]={0,0};
float sA[4]={1,0,0,1},sC[4]={0};double dA[4]={1,0,0,1},dC[4]={0};
float complex cA[4]={1,0,0,1},cC[4]={0};double complex zA[4]={1,0,0,1},zC[4]={0};
cblas_ssyrk(CblasRowMajor,CblasUpper,CblasNoTrans,n,n,sa,sA,n,sb,sC,n);CHECK("ssyrk",fabsf(sC[0]-1)<1e-4);
cblas_dsyrk(CblasRowMajor,CblasUpper,CblasNoTrans,n,n,da,dA,n,db,dC,n);CHECK("dsyrk",fabs(dC[0]-1)<EPS);
cblas_csyrk(CblasRowMajor,CblasUpper,CblasNoTrans,n,n,ca,cA,n,cb,cC,n);CHECK("csyrk",cabsf(cC[0]-1)<1e-4);
cblas_zsyrk(CblasRowMajor,CblasUpper,CblasNoTrans,n,n,za,zA,n,zb,zC,n);CHECK("zsyrk",cabs(zC[0]-1)<EPS);
cblas_cherk(CblasRowMajor,CblasUpper,CblasNoTrans,n,n,sa,cA,n,sb,cC,n);CHECK("cherk",cabsf(cC[0]-1)<1e-4);
cblas_zherk(CblasRowMajor,CblasUpper,CblasNoTrans,n,n,da,zA,n,db,zC,n);CHECK("zherk",cabs(zC[0]-1)<EPS);}
void test_syr2k_her2k(){int n=2;float sa=1,sb=0;double da=1,db=0;float ca[2]={1,0},cb[2]={0,0};double za[2]={1,0},zb[2]={0,0};
float sA[4]={1,0,0,1},sB[4]={1,0,0,1},sC[4]={0};double dA[4]={1,0,0,1},dB[4]={1,0,0,1},dC[4]={0};
float complex cA[4]={1,0,0,1},cB[4]={1,0,0,1},cC[4]={0};double complex zA[4]={1,0,0,1},zB[4]={1,0,0,1},zC[4]={0};
cblas_ssyr2k(CblasRowMajor,CblasUpper,CblasNoTrans,n,n,sa,sA,n,sB,n,sb,sC,n);CHECK("ssyr2k",fabsf(sC[0]-2)<1e-4);
cblas_dsyr2k(CblasRowMajor,CblasUpper,CblasNoTrans,n,n,da,dA,n,dB,n,db,dC,n);CHECK("dsyr2k",fabs(dC[0]-2)<EPS);
cblas_csyr2k(CblasRowMajor,CblasUpper,CblasNoTrans,n,n,ca,cA,n,cB,n,cb,cC,n);CHECK("csyr2k",cabsf(cC[0]-2)<1e-4);
cblas_zsyr2k(CblasRowMajor,CblasUpper,CblasNoTrans,n,n,za,zA,n,zB,n,zb,zC,n);CHECK("zsyr2k",cabs(zC[0]-2)<EPS);
cblas_cher2k(CblasRowMajor,CblasUpper,CblasNoTrans,n,n,ca,cA,n,cB,n,sb,cC,n);CHECK("cher2k",cabsf(cC[0]-2)<1e-4);
cblas_zher2k(CblasRowMajor,CblasUpper,CblasNoTrans,n,n,za,zA,n,zB,n,db,zC,n);CHECK("zher2k",cabs(zC[0]-2)<EPS);}
void test_trmm_trsm(){int n=2;float sa=1;double da=1;float ca[2]={1,0};double za[2]={1,0};
float sA[4]={1,0,0,1},sB[4]={2,0,0,2};double dA[4]={1,0,0,1},dB[4]={2,0,0,2};
float complex cA[4]={1,0,0,1},cB[4]={2,0,0,2};double complex zA[4]={1,0,0,1},zB[4]={2,0,0,2};
cblas_strmm(CblasRowMajor,CblasLeft,CblasUpper,CblasNoTrans,CblasNonUnit,n,n,sa,sA,n,sB,n);CHECK("strmm",fabsf(sB[0]-2)<1e-4);
cblas_dtrmm(CblasRowMajor,CblasLeft,CblasUpper,CblasNoTrans,CblasNonUnit,n,n,da,dA,n,dB,n);CHECK("dtrmm",fabs(dB[0]-2)<EPS);
cblas_ctrmm(CblasRowMajor,CblasLeft,CblasUpper,CblasNoTrans,CblasNonUnit,n,n,ca,cA,n,cB,n);CHECK("ctrmm",cabsf(cB[0]-2)<1e-4);
cblas_ztrmm(CblasRowMajor,CblasLeft,CblasUpper,CblasNoTrans,CblasNonUnit,n,n,za,zA,n,zB,n);CHECK("ztrmm",cabs(zB[0]-2)<EPS);
cblas_strsm(CblasRowMajor,CblasLeft,CblasUpper,CblasNoTrans,CblasNonUnit,n,n,sa,sA,n,sB,n);CHECK("strsm",fabsf(sB[0]-2)<1e-4);
cblas_dtrsm(CblasRowMajor,CblasLeft,CblasUpper,CblasNoTrans,CblasNonUnit,n,n,da,dA,n,dB,n);CHECK("dtrsm",fabs(dB[0]-2)<EPS);
cblas_ctrsm(CblasRowMajor,CblasLeft,CblasUpper,CblasNoTrans,CblasNonUnit,n,n,ca,cA,n,cB,n);CHECK("ctrsm",cabsf(cB[0]-2)<1e-4);
cblas_ztrsm(CblasRowMajor,CblasLeft,CblasUpper,CblasNoTrans,CblasNonUnit,n,n,za,zA,n,zB,n);CHECK("ztrsm",cabs(zB[0]-2)<EPS);}
int main(){printf("LEVEL 3 BLAS TESTS\n\n");test_gemm();test_symm_hemm();test_syrk_herk();test_syr2k_her2k();test_trmm_trsm();
printf("\nOVERALL STATUS: %s\n",ok?"PASSED":"FAILED");return ok?0:1;}