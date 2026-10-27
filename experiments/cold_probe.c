#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>

static long long mono_us(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (long long)t.tv_sec*1000000+t.tv_nsec/1000; }
static void inspect_path(const char *path) {
    const char *target=getenv("LOADING_PROBE_TARGET");
    if (!target || !path || strcmp(path,target)) return;
    int saved_errno=errno;
    long long begin=mono_us(), resident=-1, pages=-1, bytes=-1;
    int fd=syscall(SYS_openat,AT_FDCWD,path,O_RDONLY|O_CLOEXEC,0);
    struct stat st;
    if(fd>=0 && !fstat(fd,&st) && st.st_size>0) {
        bytes=st.st_size;
        long page=sysconf(_SC_PAGESIZE); pages=(bytes+page-1)/page;
        void *ptr=mmap(NULL,bytes,PROT_READ,MAP_PRIVATE,fd,0);
        unsigned char *vec=malloc(pages);
        if(ptr!=MAP_FAILED && vec && !mincore(ptr,bytes,vec)) {
            resident=0; for(long long i=0;i<pages;i++) resident+=vec[i]&1;
        }
        if(ptr!=MAP_FAILED) munmap(ptr,bytes);
        free(vec);
    }
    if(fd>=0) close(fd);
    long long end=mono_us();
    fprintf(stderr,"{\"cold_probe\":\"before_fopen\",\"mono_us\":%lld,\"probe_begin_us\":%lld,\"probe_cost_us\":%lld,\"resident_pages\":%lld,\"total_pages\":%lld,\"file_bytes\":%lld,\"tid\":%ld}\n",end,begin,end-begin,resident,pages,bytes,(long)syscall(SYS_gettid));
    errno=saved_errno;
}
FILE *fopen(const char *path,const char *mode) {
    FILE *(*real)(const char*,const char*)=dlsym(RTLD_NEXT,"fopen");
    inspect_path(path); return real(path,mode);
}
FILE *fopen64(const char *path,const char *mode) {
    FILE *(*real)(const char*,const char*)=dlsym(RTLD_NEXT,"fopen64");
    inspect_path(path); return real(path,mode);
}
