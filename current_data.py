import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
import math
from windrose import WindroseAxes
import matplotlib.gridspec as gridspec
import matplotlib as matplotlib

def get_date_indexed_df(df):
    """ (df) -> df
    
    Get timedate column for creating a timedateindex from first 6 columns of
       df.
    """
    
    date_time_columns = df[df.columns[0:6]]
    date_time_columns.columns = ['Year','Month','Day','Hour','Minute','Second']
    df.index = pd.to_datetime(date_time_columns)
    df.index.name = 'Date'
    return df

def names_as_cells(df, file, deployment):
    """ (df, str) -> df
    
    Imports a raw dataframe and renames the columns in the format 
    "cell_number_magnitude" as they increase in cell number. 
    Inputs are dataframe where columns are to be renamed and a string 
    containing the name of the file (e.g. 'magnitude' or
    'direction') which will be incorporated into the column name.
    
    Prints number of cells in deployment and df.shape.
    """
    number_of_cells = df.shape[1]
    #create a list of numbers with the correct number of cells
    column_names = list(range(1, (number_of_cells+1), 1)) 
    #empty list for new column names
    column_names_ascells_df = [] 
    
    for number in column_names:
        #create new list with format 'cell_number'
        column_names_ascells_df.append('cell' + '_' + str(number) + '_' + file) 
                                    
    #assign columns names to list column_names_ascells
    df.columns = column_names_ascells_df
    
    print('Deployment', deployment, 'has number of cells:', df.shape[1], 
          '; shape: ', df.shape)  
    
    return df
    
def plot_velocity_scatter(deployment, df_east, df_north, selected_cells, subset='', 
                          xmin=-0.5, xmax=0.5, ymin=-0.5, ymax=0.5):
    """ (str, df, df, lst[int], str, int, int, int, int) -> fig
    
    deployment (str) is defined as ID_sitename
    velocity_east_df, velocity_north_df are dataframes containing velocity data
    the list of selected_cells [int,int,int] is defined at the beginning of the 
    notebook string defining whether the data is pre/post subset (e.g. 'presubset' 
    or 'postsubset'), to be used in titles four int variables defining max and 
    min x and y limits for tht plot, defaults are set to -0.5 and 0.5
    
    Function plots a scatter of each selected cell (surface, middle and bottom) 
    with a crosshair at where x and y equal 0. Plot has a title relevant to the 
    deployment and whether the data is pre or post subset. Plot is saved as a png in 
    deployment_folder.
    """
    markersize = 5  
    fig = plt.figure(figsize=(10,10))
    plt.scatter(df_east[str("cell_{}_velocity".format(selected_cells[2]))], 
                df_north[str("cell_{}_velocity".format(selected_cells[2]))], 
                label = "Surface Cell " + str(selected_cells[2]), alpha = 0.6, 
                s= markersize)
    plt.scatter(df_east[str("cell_{}_velocity".format(selected_cells[1]))], 
                df_north[str("cell_{}_velocity".format(selected_cells[1]))], 
                label = "Middle Cell " + str(selected_cells[1]), alpha = 0.6, 
                s = markersize)
    plt.scatter(df_east[str("cell_{}_velocity".format(selected_cells[0]))], 
                df_north[str("cell_{}_velocity".format(selected_cells[0]))], 
                label = "Bottom Cell " + str(selected_cells[0]), alpha = 0.6, 
                s = markersize)
    plt.axhline(y=0, color = 'black', linestyle='-')
    plt.axvline(x=0, color = 'black', linestyle='-')
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel('East (m/s)')
    plt.ylabel('North (m/s)')
    plt.legend()
    plt.grid(True)
    plt.title('{0} Velocity {1}'.format(deployment, subset))

    plt.savefig('{0}_velocity_{1}.png'.format(deployment, subset))
    return fig

def plot_intensity_curve(deployment, file_dict, intensity_df, cell_range_list, 
                        subset='', selected_cells=None):
    """ (str, lst[str], lst[int], df, lst[int], str) -> fig
    Using the number of headers present at the top of the file, categorise
    the intensity_df by header row. Groupby category (i.e. by beam) and mean 
    across timeseries. Traspose intensity_df so columns are category and index
    is cell_range_list. Plot intensity curves and export png to current working
    directory.
    """ 

    intensity_df = categorise_df_w_multiple_headers_by_row(intensity_df, 
                                                           file_dict['intensity'])
    intensity_df = intensity_df.groupby('Category').mean()
    intensity_df = intensity_df.transpose()
    intensity_df.index = cell_range_list
    
    categories = ['beam1', 'beam2', 'beam3', 'beam4', 'average']
    
    fig = plt.figure(figsize=(10,10))
    
    for category in categories:
        plt.plot(intensity_df[category], intensity_df.index)
        
    plt.xlabel('Intensity (Counts)')
    plt.ylabel('Cell Number')
    plt.title("{0} Intensity ({1})".format(deployment, subset))
    plt.legend(categories)
    plt.xlim(40, 200)
    plt.grid(axis = 'y')
    plt.yticks(range(1, (max(cell_range_list)+1), 1))
    
    if selected_cells != None:
        plt.axhline(y = selected_cells[1], color = 'k')
        plt.axhline(y = selected_cells[2], color = 'k')
        plt.axhline(y = selected_cells[0], color = 'k')
    
    plt.savefig("{0}_intensity_{1}.png".format(deployment, subset))
    
    return fig
    
    
def plot_pgood(deployment, df, selected_cells, subset = ''):
    """ (str, df, lst[int], str) -> fig
    
    deployment (str) is defined as ID_sitename
    final percent good dataframe 
    selected_cells list[int] defined at the start of notebook """ 

    #plot total %good data 
    fig, (ax1, ax2, ax3) = plt.subplots(nrows = 3, ncols = 1, sharex = True, 
                                        figsize=(10,10))
    ax1.plot(df[str("cell_{}_percentgood".format(selected_cells[2]))], 
                                                 linestyle = ' ', marker = '.')
    ax1.set(title = "{0} Total % Good ({1})".format(deployment, subset), 
            ylabel = str("cell {}".format(selected_cells[2])))
    ax1.grid()

    ax2.plot(df[str("cell_{}_percentgood".format(selected_cells[1]))], 
                                                 linestyle = ' ', marker = '.')
    ax2.set(ylabel = str("cell {}".format(selected_cells[1])))
    ax2.grid()

    ax3.plot(df[str("cell_{}_percentgood".format(selected_cells[0]))], 
                                                 linestyle = ' ', marker = '.')
    ax3.set(ylabel = str("cell {}".format(selected_cells[0])))
    ax3.grid()

    plt.savefig("{}_percentgood_selected_cells.png".format(deployment))
    return fig


def plot_summary_data(deployment, headingpitchroll_df, summary_data_df, subset = ''):
    """ (str, df, df, str) -> fig
    
    deployment (str) is defined as ID_sitename
    headingpitchroll_df created in notebook
    summary_data_df created in notebook
    
    plot summary data in 4 windows (pitch, roll, heading, total depth) across
    the whole deployment.
    """

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows = 4, ncols = 1, sharex = True, 
                                             figsize=(10,10))
    ax1.plot(headingpitchroll_df.Pitch)
    ax1.set(title = "{0} Summary Data ({1})".format(deployment, subset), 
            ylabel = 'Pitch')
    ax1.grid()

    ax2.plot(headingpitchroll_df.Roll)
    ax2.set(ylabel = 'Roll')
    ax2.grid()

    ax3.plot(headingpitchroll_df.Heading)
    ax3.set(ylabel = 'Heading')
    ax3.grid()

    ax4.plot(summary_data_df.totaldepth)
    ax4.set(ylabel = "Total Depth", xlabel = 'Num')
    ax4.grid()

    plt.savefig("{}_pitch_roll_heading_sensordepth_presubset.png".format(deployment))
    return fig


def drop_metadata_cols_and_dups(data_df):
    """ (df) -> df
    
    Drop the first 6 columns from the dataframe after using get_date_indexed_df(),
    making these columns redundant. Removes any duplicate rows from data_df. 
    """
    
    data_df = data_df.drop_duplicates()
    data_df = data_df.drop([1,2,3,4,5,6], axis = 1)
    
    return data_df
    

def subset_df(df, start_date, end_date):
    """  (df, datetime, datetime) -> df
    
    Subsets datetimeindexed df based on start_date and
    end_date. 
    """
    
    df_subset = df[start_date : end_date]
    
    return df_subset 


def how_many_headers_txt(filename):
    """ (str) -> int
    Reads in data filename by row and counts how many 
    start with 'Num'. Function assumes that all header lines
    begin with 'Num'.
    """
    
    f = open(filename)
    csv_f = csv.reader(f)
    
    headers = []
    for row in csv_f:
        if 'Num' in row[0]:
            headers.append(row)
    number_of_headers = len(headers)
    
    return number_of_headers  


def import_data_txt(filename):
    """ (str) -> df
    Reads in df using str filename and skips header rows. Removes all duplicate 
    rows. 
    """

    number_of_headers = how_many_headers_txt(filename)
    print('Number of header lines: ', number_of_headers)
    
    header_range_list = list(range(0, (number_of_headers)))
    
    data_df = pd.read_csv(filename, skiprows = header_range_list, header = None)
    data_df = data_df.drop([0], axis = 1)
    
    return data_df
    
def selected_cells_str_list(selected_cells, dataset_name):
    """ (list[int], str) -> list[str]
    Creates list of strings with cell_number_nameofdataset format (e.g.
    'cell_5_percentgood') for selecting columns in a dataframe. 
    """
    strings = []
    
    for cell in range(len(selected_cells)):
        strings.append(str("cell_{0}_{1}".format(selected_cells[cell], dataset_name)))
    
    return strings
    
def categorise_df_w_multiple_headers_by_row(data_df, filename):
    """ (df, str) -> df
    Using number of headers in input txt file, create categorical variable column
    identifies data by header.
    """
    number_of_headers = how_many_headers_txt(filename)
    number_of_rows = int(data_df.shape[0]/number_of_headers)
    
    Categories = ['beam1', 'beam2', 'beam3', 'beam4', 'average'] * number_of_rows
    data_df['Category'] = Categories
    
    return data_df 

def ping_calcs(selectedcells_percentgood_4beam, selectedcells_percentgood_3beam, 
               selected_cells, percentgood_strings):
    """ df, df, list[int], list[str] -> df
    Sum percentgood3beam and percentgood3beam columns for selected cells. Create
    list of totalvalidpings_strings and stdev_strings for selecting data columns.
    Carry out calculations to create total valid pings and standard deviation 
    columns in final_percentgood_df.
    """
    final_percentgood_df = selectedcells_percentgood_4beam.add(selectedcells_percentgood_3beam, 
                                                               fill_value=0)
    
    totalvalidpings_strings = selected_cells_str_list(selected_cells, 
                                                                   'totalvalidpings')
    stdev_strings = selected_cells_str_list(selected_cells, 'stdev')
    
    for cell in range(len(selected_cells)):
        final_percentgood_df[totalvalidpings_strings[cell]] = (final_percentgood_df[percentgood_strings[cell]]/ 100) * 300
    for cell in range(len(selected_cells)):
        final_percentgood_df[stdev_strings[cell]] = 10.9/np.sqrt(final_percentgood_df[totalvalidpings_strings[cell]])
    
    return final_percentgood_df
    
def circular_mean(weights, angles):
    """ (df, df) -> series
    
    Mathematical method for creating meaningful weighted mean direction with
    respect to magnitude. df's  """
    
    x = y = 0.
    for angle, weight in zip(angles, weights):
        x += math.cos(math.radians(angle)) * weight
        y += math.sin(math.radians(angle)) * weight

    mean = math.degrees(math.atan2(y, x))
    if mean < 0:
        mean = 360 + mean
    return mean 
    
def plot_mag_vel_graphs(deployment, magnitude_mean_df, direction_mean_df,
    no_nan_dir_mag_df, water_cell_range_list, mag_dir_cells_df, selected_cells,
    velocity_north_df, velocity_east_df, mean_df, subset_title = '', subset = ''):
    """ (str, df, df, df, lst[int], df, lst[int], df, df, str, str) -> fig, fig, fig, 
    fig
                                                                                                                                 
    deployment (str) is defined as ID_sitename
    The magnitude_mean_df is defined in the notebook                                                                            
    The direction_mean_df is defined in the notebook
    The no_nan_dir_mag_df is defined in the notebook
    list[int] for water_cell_range_list (y axis for plotting) defined in
    notebook
    The mag_dir_cells_df is defined in the notebook
    The list of selected_cells [int,int,int] is defined at the beginning of the
    notebook 
    string defining whether the data is pre/post subset (e.g. 'presubset' or
    'postsubset') to be used in titles
    velocity_east_df is defined earlier in the notebook"""
    #mean dir, mag, nan
    matplotlib.rcParams.update({'font.size': 15})
    fig = plt.figure(figsize = (10,15))

    gs1 = gridspec.GridSpec(3, 3, wspace=0.4)

    ax1 = plt.subplot(gs1[:,:-1])
    ax1.plot(magnitude_mean_df.mean_magnitude, water_cell_range_list, color =
    'b')
    ax1.yaxis.set_ticks(range(1, (max(water_cell_range_list)+1), 1))
    ax1.grid(axis = 'y')
    ax1.set_xlabel('mean magnitude (m/sec)', color = 'b')
    ax1.set_ylabel('cell number')
    ax1.tick_params('x', colors='b')
    for index in range(3):
        plt.axhline(y = selected_cells[index], color = 'k')

    ax2 = plt.subplot(gs1[:,:-1])
    ax2 = ax1.twiny()
    ax2.plot(direction_mean_df, water_cell_range_list, color = 'g')
    ax2.set_xlabel('mean direction (deg mag)', color = 'g')
    ax2.tick_params('x', colors='g')
    ax2.set_xticks([0,90,180,270])
    ax2.set_xlim(0,360)

    for angle in [90, 180, 270]:
        plt.axvline(x = angle, linestyle = '--', lw = '1', color = 'k')

    ax3 = plt.subplot(gs1[:, -1])
    ax3.plot(no_nan_dir_mag_df, water_cell_range_list, color = 'k')
    ax3.set_xlabel('Number of NaNs')
    ax3.yaxis.set_ticks(range(1, (max(water_cell_range_list)+1), 1))
    ax3.grid(axis = 'y')

    for index in range(3):
        plt.axhline(y = selected_cells[index], color = 'k')

    plt.savefig("{0}_Mean_Direction_and_Magnitude_{1}".format(deployment,
    subset))
    
    #mag vs time
    plt.figure(figsize=(200,10))
    plt.plot(mag_dir_cells_df[str("cell_{}_magnitude".format(selected_cells[2]))])
    plt.plot(mag_dir_cells_df[str("cell_{}_magnitude".format(selected_cells[1]))])
    plt.plot(mag_dir_cells_df[str("cell_{}_magnitude".format(selected_cells[0]))])
    plt.title("{0} Magnitude {1}".format(deployment, subset))
    plt.ylabel('Magnitude (m/s)')
    plt.xlabel('Date (YYY-MM-DD)')
    #plt.xlim(min(mag_dir_cells_df.index)-40, max(mag_dir_cells_df.index)+40) #sets the size of the blank space at the edge of the plot 
    plt.legend()
    plt.grid()
    plt.savefig("{0}_Magnitude_{1}.png".format(deployment, subset))

    #vel north and east vs time
    fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize=(100,15))

    ax1.plot(velocity_north_df[str("cell_{}_velocity".format(selected_cells[2]))])
    ax1.plot(velocity_north_df[str("cell_{}_velocity".format(selected_cells[1]))])
    ax1.plot(velocity_north_df[str("cell_{}_velocity".format(selected_cells[0]))])
    ax1.set_ylabel('velocity North (m/sec)')
    ax1.grid()
    ax1.legend()

    ax2.plot(velocity_east_df[str("cell_{}_velocity".format(selected_cells[2]))])
    ax2.plot(velocity_east_df[str("cell_{}_velocity".format(selected_cells[1]))])
    ax2.plot(velocity_east_df[str("cell_{}_velocity".format(selected_cells[0]))])
    ax2.set_ylabel('Velocity East (m/sec))')
    ax2.grid()
    ax2.legend()

    plt.xlabel('Date (YYY-MM-DD)')
    #plt.xlim(min(mag_dir_cells_df.index)-40, max(mag_dir_cells_df.index)+40) #sets the size of the blank space at the edge of the plot

    plt.suptitle("{0} Velocity North and East vs Time {1}".format(deployment, subset))
    plt.savefig("{0}_Velocity_North_and_East_vs_Time_{1}".format(deployment, subset))
    
    #current rose
    fig=plt.figure(figsize=(10,10))
    rect=[0.05,0.05,0.8,0.8] 
    wa=WindroseAxes(fig, rect)
    fig.add_axes(wa)
    wa.bar(mean_df.MeanDirection, mean_df.MeanSpeed, bins=6, normed=True, opening=0.8, edgecolor='white')
    plt.legend(fontsize = 20)
    plt.title("{0} {1} \nMean Velocity (m/Sec) - radial values indicate percentage of occurance".format(deployment, subset), size=15)
    leg = plt.legend(0)
    #Sort out units if you changed to knots
    leg.set_title('Velocity - (m/Sec)')
    plt.savefig("{}_Current_Rose_Presubset".format(deployment))
