# obj2grid
A command line tool that converts Wavefront OBJ files to Pikmin 2 grid.bin (collision) 
and mapcode.bin (footstep sounds) files, and converts Pikmin 2 grid.bin files back into Wavefront OBJ.

# Requirements
This tool requires Python 3. Newer = better (at the moment the newest version is 3.6 or so)
When installing Python on Windows, make sure you check the box that says "Add Python to PATH" 
if you want to make use of the included .bat files.

# How To
The .obj model file you want to convert needs to be triangulated (i.e. all faces are triangles and have exactly 3 vertices).
If it isn't triangulated, obj2grid will throw an error.

For a quick set-up on Windows, drag your .obj file onto the included make_collision.bat which will create
a grid.bin and a mapcode.bin out of your .obj file in the folder of the .obj file. In a similar way, drag
a grid.bin file onto make_obj.bat to turn Pikmin 2 collision back into .obj.

Information for command line usage:
python obj2grid.py [-h] [--cell_size CELL_SIZE] [--grid2obj] 
                    input [output_grid] [output_mapcode]
                        
                        
positional arguments:
  input                 Filepath of the wavefront .obj file that will be
                        converted into collision. If --grid2obj is set,
                        filepath of the grid.bin to be converted into .obj
                        
  output_grid           Output path of the created collision file. If
                        --grid2obj is set, output path of the created obj file
                        
  output_mapcode        Output path of the created mapcode file

optional arguments:
  -h, --help            show this help message and exit
  --cell_size CELL_SIZE
                        Size of cells in grid structure. Bigger can result in
                        smaller file but lower ingame performance
                        
  --grid2obj            Use this option to create an OBJ file out of a
                        grid.bin file

Examples:
python obj2grid.py --cell_size 100 MyCustomCollision.obj MyPikminGrid.bin MyPikminMapcode.bin 

python obj2grid.py --grid2obj MyPikminGrid.bin MyCollision.obj

Note: you can also put in absolute file paths (i.e. ones starting with C:\) so that your finished collision
is put directly into the folder you want them in.