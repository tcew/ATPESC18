/****************************************************************************
    png.c - read and write png images using libpng routines.
    Distributed with Xplanet.
    Copyright (C) 2002 Hari Nair <hari@alumni.caltech.edu>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
****************************************************************************/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <png.h>
//#include <png.h>

int read_png(const char *filename, int *width, int *height, unsigned char **rgb, 
	     unsigned char **alpha);

int write_png(FILE *outfile, int width, int height, unsigned char *rgb,
	      unsigned char *alpha);

int write_gray_png(FILE *outfile, int width, int height, float *img, float minI, float maxI);
int write_hot_png(FILE *outfile, int width, int height, float *img, float minI, float maxI);
