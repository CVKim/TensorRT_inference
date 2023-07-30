#pragma once
#define NONMINMAX

//#define WIN32_LEAN_AND_MEAN             // ���� ������ �ʴ� ������ Windows ������� �����մϴ�.
#define _ATL_CSTRING_EXPLICIT_CONSTRUCTORS      // �Ϻ� CString �����ڴ� ��������� ����˴ϴ�.

//#include <afxmt.h> 
#include <Shlwapi.h>

#include <chrono>
#include <intsafe.h>

#define M_PI	3.14159265359
#define Duration(a) std::chrono::duration_cast<std::chrono::milliseconds>(a).count()
#define timeNow() std::chrono::high_resolution_clock::now