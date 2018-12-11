NVCC = nvcc
NVCCFLAGS = -Wno-deprecated-gpu-targets
CUSPARSE = -lcusparse
NV_PROGS = cluster.exe
OBJDIR = objects
.PHONY: all clean

all: $(OBJDIR) $(NV_PROGS)

clean:
	rm -f *.o $(NV_PROGS) $(OBJDIR)/*.o

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(NV_PROGS): $(OBJDIR)/cluster.o $(OBJDIR)/fileUtils.o $(OBJDIR)/testUtils.o $(OBJDIR)/deviceUtils.o $(OBJDIR)/mathUtils.o $(OBJDIR)/kernelUtils.o
	$(NVCC) $(OBJDIR)/cluster.o $(OBJDIR)/fileUtils.o $(OBJDIR)/testUtils.o $(OBJDIR)/deviceUtils.o $(OBJDIR)/mathUtils.o $(OBJDIR)/kernelUtils.o $(CUSPARSE) $(NVCCFLAGS) -o $(NV_PROGS)

$(OBJDIR)/kernelUtils.o: kernelUtils.cu
	$(NVCC) -c kernelUtils.cu $(NVCCFLAGS) -o $(OBJDIR)/kernelUtils.o

$(OBJDIR)/mathUtils.o: mathUtils.cu
	$(NVCC) -c mathUtils.cu $(NVCCFLAGS) -o $(OBJDIR)/mathUtils.o

$(OBJDIR)/deviceUtils.o: deviceUtils.cu
	$(NVCC) -c deviceUtils.cu $(NVCCFLAGS) -o $(OBJDIR)/deviceUtils.o

$(OBJDIR)/cluster.o: cluster.cpp
	$(NVCC) -c cluster.cpp $(NVCCFLAGS) -o $(OBJDIR)/cluster.o

$(OBJDIR)/fileUtils.o: fileUtils.cpp
	$(NVCC) -c fileUtils.cpp $(NVCCFLAGS) -o $(OBJDIR)/fileUtils.o

$(OBJDIR)/testUtils.o: testUtils.cpp
	$(NVCC) -c testUtils.cpp $(NVCCFLAGS) -o $(OBJDIR)/testUtils.o