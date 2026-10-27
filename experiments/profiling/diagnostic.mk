include Makefile
CXXFLAGS += -I/usr/local/cuda/include
LDFLAGS += -L/usr/local/cuda/lib64 -lnvToolsExt -lcudart
lora-cost-probe: probe.cpp $(OBJ_ALL)
	$(CXX) $(CXXFLAGS) -Icommon -c $< -o lora-cost-probe.o
	$(CXX) $(CXXFLAGS) $(OBJ_ALL) lora-cost-probe.o -o $@ $(LDFLAGS) -L/usr/local/cuda/lib64 -lnvToolsExt -lcudart
