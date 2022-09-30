# cv-traffic
Computer vision for vehicle tracking and classification

# Project structure

La estructura actual se basa en [Good Enough Project](https://github.com/bvreede/good-enough-project/blob/master/README.md), [Reproducible Science Cookiecutter](https://github.com/mkrapp/cookiecutter-reproducible-science) por [Mario Krapp](https://github.com/mkrapp), [Cookiecutter Data Science](https://github.com/drivendata/cookiecutter-data-science): *A logical, reasonably standardized, but flexible project structure for doing and sharing data science work* y [Good Enough Practices in Scientific Computing](https://doi.org/10.1371/journal.pcbi.1005510), Wilson _et al._, PLOS Computational Biology (2017).

La estructura del Proyecto considera tres tipos de carpetas:
- read-only (RO): archivos no editados ni por el código ni por los responsables del análisis.
- human-writeable (HW): archivos editados únicamente por los responsables del análisis.
- project-generated (PG): carpetas generadas al ejecutar el código. Estas carpetas pueden ser borradas, y serán recreadas al correrse el Proyecto.

```
.
├── .gitignore
├── CITATION.md
├── LICENSE.md
├── README.md
├── requirements.txt   <- Requisitos del análisis
├── bin                <- Código compilado y código externo, ignorado por git. (PG)
│   └── external       <- Cualquier código externo ignorado por git. (RO)
├── config             <- Archivos de configuración. (HW)
├── commons            <- Archivos compartidos que no forman parte del análisis, pero que se utilian para la generación de reportes. (HW)
├── data               <- Data del proyecto, ignorada por git.
│   ├── processed      <- Data final empleada en los análisis. (PG)
│   ├── raw            <- Data original, no puede ser manipulada. (RO)
│   └── temp           <- Data intermedia que ha sido transformada. (PG)
├── output             <- Resultados del análisis
└── scripts            <- Código fuente del Proyecto (HW)

```

# MOT Architectures

1. Python: Real-time Multiple Object Tracking (MOT) with Yolov3, Tensorflow and Deep SORT

https://www.youtube.com/watch?list=PLamezrmCJvNFtrTQ8mvPIbLrjNp_GnOhC&v=zi-62z-3c4U&feature=youtu.be

Please visit https://www.youtube.com/watch?v=zi-62z-3c4U for the full course - Real-time Multiple Object Tracking (MOT) with Yolov3, Tensorflow and Deep SORT

2. Object Tracking with OpenCV

https://livecodestream.dev/post/object-tracking-with-opencv/
https://broutonlab.com/blog/opencv-object-tracking

3. Vehicle Counting, classification & Detection using OpenCV & Python

https://techvidvan.com/tutorials/opencv-vehicle-detection-classification-counting/

4. Build your own Vehicle Detection Model using OpenCV and Python (2020)

https://www.analyticsvidhya.com/blog/2020/04/vehicle-detection-opencv-python/

5. Building Vehicle Counter System Using OpenCV - Eucledian Distance Tracker

https://www.analyticsvidhya.com/blog/2022/04/building-vehicle-counter-system-using-opencv/

6. Vehicle Detection and Counting System using OpenCV

https://www.analyticsvidhya.com/blog/2021/12/vehicle-detection-and-counting-system-using-opencv/

7. OpenCV for Beginners

Course
https://opencv.org/multiple-object-tracking-in-realtime/