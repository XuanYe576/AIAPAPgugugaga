clear all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%% Variables entered by user %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x_grid_div = 10; %%%%% Number of squares on grid in the x-axis
y_grid_div = 10; %%%%% Number of squares on grid in the y-axis
start_patch = 1001;
stop_patch = 6000;
num_arr = (stop_patch-start_patch)+1;
grid_sq_dim = 10; %%%%% The size of each squre on the grid in mm

feed_x_dim = 2.96; %%%%% The width of the feeding port strip in mm
feed_y_dim = 25; %%%%% The length of the feeding port strip in mm

substrate_x_dim = 150; %%%%% Width of substrate in mm
substrate_y_dim = 150; %%%%% Length of substrate in mm

substrate_h = 1.57; %%%%% The thickness of the substrate in mm
substrate_er = 4.4; %%%%% Relative permittivity of the substrate

ground_x_dim = substrate_x_dim; %%%%% Width of ground plate in mm
ground_y_dim = substrate_y_dim; %%%%% Length of ground plate in mm

patch_offset_x = 25; %%%%% Distance away from left edge of substrate in mm
patch_offset_y = 25; %%%%% Distance away from bottom edge of substrate in mm

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



patch_ant = randi([0,1], x_grid_div, y_grid_div, num_arr); %%%%% Creates num_arr arrays that are x_grid_div by y_grid_div

writecell({'Number of antennas',num_arr},'patch_antennas_1001_to_6000.csv','WriteMode','append');
writecell({'number of divisions in x axis',x_grid_div},'patch_antennas_1001_to_6000.csv','WriteMode','append');
writecell({'number of divisions in y axis',x_grid_div},'patch_antennas_1001_to_6000.csv','WriteMode','append');


for i = 1:1:num_arr
    a_str=strcat('Patch Antenna', num2str(i+(start_patch - 1)));
    %%T = table(a_str);
    writecell({a_str},'patch_antennas_1001_to_6000.csv','WriteMode','append');
    writematrix(patch_ant(:,:,i),'patch_antennas_1001_to_6000.csv','WriteMode','append')

end


