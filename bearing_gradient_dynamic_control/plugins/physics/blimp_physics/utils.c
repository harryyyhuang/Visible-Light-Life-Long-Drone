/****************************************************************************

blimp_physics -- A blimp physics model for Webots.

Copyright (C) 2006 Laboratory of Intelligent Systems, EPFL, Lausanne
Authors:    Alexis Guanella            guanella@ini.phys.ethz.ch
            Antoine Beyeler            antoine.beyeler@epfl.ch
            Jean-Christophe Zufferey   jean-christophe.zufferey@epfl.ch
            Dario Floreano             dario.floreano@epfl.ch
Web: http://lis.epfl.ch

The authors of any publication arising from research using this software are
kindly requested to add the following reference:

        Zufferey, J.C., Guanella, A., Beyeler, A., Floreano, D. (2006) Flying over
        the Reality Gap: From Simulated to Real Indoor Airships. Autonomous Robots,
        Springer US.

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
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

******************************************************************************/
/*------------------------------------------------------------------------------

Author:		Alexis Guanella (ag)
                        Antoine Beyeler (ab)

------------------------------------------------------------------------------*/

#include "utils.h"

void utils_SetZero(dReal *a, int n) {
  while (n > 0) {
    *(a++) = 0;
    n--;
  }
}

void utils_SetValue(dReal *a, int n, dReal value) {
  while (n > 0) {
    *(a++) = value;
    n--;
  }
}
void utils_Assign(dReal *a, const dReal *b, int n) {
  while (n > 0) {
    *(a++) = *(b++);
    n--;
  }
}
void utils_Add(dReal *a, const dReal *b, int n) {
  while (n > 0) {
    *(a++) += *(b++);
    n--;
  }
}

void utils_Multiply(dReal *A, const dReal *B, const dReal *C, int p, int q, int r) {
  int i, j, k, qskip, rskip;
  dReal sum;
  const dReal *b, *c, *bb;
  qskip = q;
  rskip = r;

  bb = B;
  for (i = p; i; i--) {
    for (j = 0; j < r; j++) {
      c = C + j;
      b = bb;
      sum = 0;
      for (k = q; k; k--, c += rskip)
        sum += (*(b++)) * (*c);
      *(A++) = sum;
    }
    bb += qskip;
  }
}

void utils_InvertMatrix33(dReal *m) {
  dReal determinant =
    m[0] * m[4] * m[8] + m[1] * m[5] * m[6] + m[2] * m[3] * m[7] - m[6] * m[4] * m[2] - m[7] * m[5] * m[0] - m[8] * m[3] * m[1];

  dReal tmp[9];

  tmp[0] = (m[4] * m[8] - m[5] * m[7]) / determinant;
  tmp[1] = (m[2] * m[7] - m[1] * m[8]) / determinant;
  tmp[2] = (m[1] * m[5] - m[2] * m[4]) / determinant;
  tmp[3] = (m[5] * m[6] - m[3] * m[8]) / determinant;
  tmp[4] = (m[0] * m[8] - m[2] * m[6]) / determinant;
  tmp[5] = (m[2] * m[3] - m[0] * m[5]) / determinant;
  tmp[6] = (m[3] * m[7] - m[4] * m[6]) / determinant;
  tmp[7] = (m[1] * m[6] - m[0] * m[7]) / determinant;
  tmp[8] = (m[0] * m[4] - m[1] * m[3]) / determinant;

  utils_Assign(m, tmp, 9);
}

/* Cross-platform debug logging implementation */
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

#ifdef _WIN32
  #include <windows.h>
  #include <libloaderapi.h>
#elif defined(__APPLE__)
  #include <dlfcn.h>
  #include <libgen.h>
#elif defined(__linux__)
  #define _GNU_SOURCE
  #include <dlfcn.h>
  #include <libgen.h>
#endif

const char* utils_GetPluginLogPath(const char* filename) {
    static char log_path[512] = {0};
    static char cached_filename[256] = {0};
    
    // If filename changed or first call, recalculate path
    if (log_path[0] == '\0' || strcmp(cached_filename, filename) != 0) {
        char dll_path[512] = {0};
        
        // Cache the filename for future comparisons
        strncpy(cached_filename, filename, sizeof(cached_filename)-1);
        
#ifdef _WIN32
        // Windows: Get DLL path using GetModuleFileName
        HMODULE hm = NULL;
        if (GetModuleHandleEx(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | 
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              (LPCSTR)&utils_GetPluginLogPath, &hm)) {
            GetModuleFileName(hm, dll_path, sizeof(dll_path));
            
            // Find last backslash and replace with logs path
            char* last_slash = strrchr(dll_path, '\\');
            if (last_slash) {
                *last_slash = '\0';
                snprintf(log_path, sizeof(log_path), "%s\\logs\\%s", dll_path, filename);
            }
        }
        
#elif defined(__APPLE__) || defined(__linux__)
        // Unix-like: Get shared library path using dladdr
        Dl_info dl_info;
        if (dladdr((void*)utils_GetPluginLogPath, &dl_info)) {
            strncpy(dll_path, dl_info.dli_fname, sizeof(dll_path)-1);
            
            // Get directory and append logs path
            char* dir = dirname(dll_path);
            snprintf(log_path, sizeof(log_path), "%s/logs/%s", dir, filename);
        }
#endif
        
        // Fallback if platform-specific method fails
        if (log_path[0] == '\0') {
            snprintf(log_path, sizeof(log_path), "logs/%s", filename);
        }
        
        // Debug: Write path info to a debug file (only on first call)
        static int debug_written = 0;
        if (!debug_written) {
            FILE *debug_file = fopen("debug_log_path.txt", "w");
            if (debug_file) {
                fprintf(debug_file, "Platform: ");
#ifdef _WIN32
                fprintf(debug_file, "Windows\n");
#elif defined(__APPLE__)
                fprintf(debug_file, "macOS\n");
#elif defined(__linux__)
                fprintf(debug_file, "Linux\n");
#else
                fprintf(debug_file, "Unknown\n");
#endif
                fprintf(debug_file, "DLL/SO path: %s\n", dll_path);
                fprintf(debug_file, "Log path template: [dll_dir]/logs/[filename]\n");
                fprintf(debug_file, "Example log path: %s\n", log_path);
                fclose(debug_file);
            }
            debug_written = 1;
        }
    }
    
    return log_path;
}

void utils_DebugLog(const char* filename, const char* format, ...) {
    FILE *debug_fp = fopen(utils_GetPluginLogPath(filename), "a");
    if (debug_fp) {
        va_list args;
        va_start(args, format);
        vfprintf(debug_fp, format, args);
        va_end(args);
        fclose(debug_fp);
    }
}
