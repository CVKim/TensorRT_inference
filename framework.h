#pragma once
#define NONMINMAX

//#define WIN32_LEAN_AND_MEAN             // 거의 사용되지 않는 내용을 Windows 헤더에서 제외합니다.
#define _ATL_CSTRING_EXPLICIT_CONSTRUCTORS      // 일부 CString 생성자는 명시적으로 선언됩니다.

//#include <afxmt.h> 
#include <Shlwapi.h>

#include <chrono>
#include <intsafe.h>

#define M_PI	3.14159265359
#define Duration(a) std::chrono::duration_cast<std::chrono::milliseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now