
#ifndef PTI_EXPORT_H
#define PTI_EXPORT_H

#ifdef PTI_STATIC_DEFINE
#  define PTI_EXPORT
#  define PTI_NO_EXPORT
#else

#  ifdef WIN32
     /* Windows (MSVC/MinGW) */
#    ifndef PTI_EXPORT
#      ifdef pti_EXPORTS
         /* We are building this library */
#        define PTI_EXPORT __declspec(dllexport)
#      else
         /* We are using this library */
#        define PTI_EXPORT __declspec(dllimport)
#      endif
#    endif

#    ifndef PTI_NO_EXPORT
#      define PTI_NO_EXPORT
#    endif

#  else

     /* Linux / Unix — GCC/Clang visibility */
#    ifndef PTI_EXPORT
#      ifdef pti_EXPORTS
         /* We are building this library */
#        define PTI_EXPORT __attribute__((visibility("default")))
#      else
         /* We are using this library */
#        define PTI_EXPORT __attribute__((visibility("default")))
#      endif
#    endif

#    ifndef PTI_NO_EXPORT
#      define PTI_NO_EXPORT __attribute__((visibility("hidden")))
#    endif

#  endif /* WIN32 */

#endif /* PTI_STATIC_DEFINE */

#ifndef PTI_DEPRECATED
#  define PTI_DEPRECATED
#endif

#ifndef PTI_DEPRECATED_EXPORT
#  define PTI_DEPRECATED_EXPORT PTI_EXPORT PTI_DEPRECATED
#endif

#ifndef PTI_DEPRECATED_NO_EXPORT
#  define PTI_DEPRECATED_NO_EXPORT PTI_NO_EXPORT PTI_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef PTI_NO_DEPRECATED
#    define PTI_NO_DEPRECATED
#  endif
#endif

#endif /* PTI_EXPORT_H */
