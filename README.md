# 3D_Voronoi_Plane_Trans-D_Inversion
This is a Python code based on MPI module that parametrizes the subsurface structures using 3D Voronoi and Plane (VP) partitioning. You can find the full description in my research paper published in Geophysical Journal International (GJI). You can freely use this code only if you cite this research paper. Citation instruction will be added soon. Enjoy!

First, design job.sh file. Since this is a MPI code, by default I choose 20 parallel processors. you can change 20.
Running job.sh will run VL_3D_MPI.py
Two text files (RTP_2D_Data.txt & GRV_2D_Data.txt) are simulated data. the first column shows X or Easting coordinates. The second column shows Y or Northing coordinates. and the third column is gravity or magnetic value that I set all 99. These 99 are not functional in this code because true model will be later generated and produces simulated data.
Enjoy!
