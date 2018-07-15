ifndef OCCA_DIR
	OCCA_DIR = ../../
endif

include ${OCCA_DIR}/scripts/makefile

#---[ COMPILATION ]-------------------------------
headers = $(wildcard $(iPath)/*.hpp) $(wildcard $(iPath)/*.tpp)
sources = $(wildcard $(sPath)/*.cpp)

objects  = $(subst $(sPath)/,$(oPath)/,$(sources:.cpp=.o))

executables: simple 

simple: $(objects) $(headers) simple.cpp
	$(compiler) $(compilerFlags) -o simple $(flags) $(objects) simple.cpp $(paths) $(links)

simpleV2: $(objects) $(headers) simpleV2.cpp
	$(compiler) $(compilerFlags) -o simpleV2 $(flags) $(objects) simpleV2.cpp $(paths) $(links)

$(oPath)/%.o:$(sPath)/%.cpp $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.hpp))) $(wildcard $(subst $(sPath)/,$(iPath)/,$(<:.cpp=.tpp)))
	$(compiler) $(compilerFlags) -o $@ $(flags) -c $(paths) $<

clean:
	rm -f $(oPath)/*;
	rm -f simpleV2 simple;
#=================================================
