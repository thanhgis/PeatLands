from GEE_wrappers import GEE_extent
import numpy as np
import os
import math
import datetime as dt

def get_map(minlon, minlat, maxlon, maxlat, outpath, sampling=30, year=None, month=None, day=None, tracknr=None, overwrite=False, start=None, stop=None):

	if year is not None:
		# initialise GEE retrieval dataset
		GEE_interface = GEE_extent(minlon, minlat, maxlon, maxlat, outpath, sampling=sampling)
		GEE_interface.init_SM_retrieval(year, month, day, track=tracknr)
		if GEE_interface.ORBIT == 'ASCENDING':	orbit_prefix = 'A'
		else:	orbit_prefix = 'D'
		outname = 'SMCS1_' + str(GEE_interface.S1_DATE.year) + '{:02d}'.format(GEE_interface.S1_DATE.month) + '{:02d}'.format(GEE_interface.S1_DATE.day) + '_' + '{:02d}'.format(GEE_interface.S1_DATE.hour) + '{:02d}'.format(GEE_interface.S1_DATE.minute) + '{:02d}'.format(GEE_interface.S1_DATE.second) + '_' + '{:03d}'.format(math.trunc(GEE_interface.TRACK_NR)) + '_' + orbit_prefix
		# Estimate soil moisture
		GEE_interface.estimate_SM_GBR_1step()
		GEE_interface = None
	else:
		# if no specific date was specified extract entire time series
		GEE_interface = GEE_extent(minlon, minlat, maxlon, maxlat, outpath, sampling=sampling)
		# get list of S1 dates
		dates, orbits = GEE_interface.get_S1_dates(tracknr=tracknr, ascending=False, start=start, stop=stop)
		dates, unqidx = np.unique(dates, return_index=True)
		orbits = orbits[unqidx]
		todeldates = list()
		for i in range(1, len(dates)):	
			if ((dates[i] - dates[i - 1]) < dt.timedelta(minutes=10)) & (orbits[i] == orbits[i - 1]):
				todeldates.append(i)
		dates = np.delete(dates, todeldates)
		orbits = np.delete(orbits, todeldates)
		for dateI, orbitI in zip(dates, orbits):
			print(dateI, orbitI)
			if GEE_interface.ORBIT == 'ASCENDING':
				orbit_prefix = 'A'
			else:
				orbit_prefix = 'D'	
			outname = 'SMCS1_' + str(GEE_interface.S1_DATE.year) + '{:02d}'.format(GEE_interface.S1_DATE.month) + '{:02d}'.format(GEE_interface.S1_DATE.day) + '_' + '{:02d}'.format(GEE_interface.S1_DATE.hour) + '{:02d}'.format(GEE_interface.S1_DATE.minute) + '{:02d}'.format(GEE_interface.S1_DATE.second) + '_' + '{:03d}'.format(math.trunc(GEE_interface.TRACK_NR)) + '_' + orbit_prefix
			if overwrite == False and os.path.exists(outpath + outname + '.tif'):
				print(outname + ' already done')
			continue
		# Estimate soil moisture	GEE_interface.estimate_SM_GBR_1step()
		if GEE_interface.ESTIMATED_SM is not None:		
			GEE_interface.GEE_2_asset(name=outname, timeout=False)
			GEE_interface = None

def get_ts(loc, workpath, tracknr=None, footprint=50, masksnow=True, calc_anomalies=False, create_plots=False, names=None):
	print (loc)
	if isinstance(loc, list):
		print('list')
	else:
		loc = [loc]

	if names is not None:
		if isinstance(names, list):	print('Name list specified')
		else:	names = [names]

	sm_ts_out = list()
	names_out = list()

	cntr = 0
	lon = loc[0]
	lat = loc[1]

	if names is not None:
		iname = names[cntr]
	else:
		iname = None

	print('Estimating surface soil moisture for lon: ' + str(lon) + ' lat: ' + str(lat))

	# initialize GEE point object
	gee_pt_obj = GEE_pt(lon, lat, workpath, buffer=footprint)
	sm_ts = gee_pt_obj.extr_SM(tracknr=None, masksnow=False, tempfilter=False, calc_anomalies=calc_anomalies)
	if iname is not None:
		# create plots
		if create_plots == True:	
			if calc_anomalies == False:		
				sm_ts_out.append(0)
		names_out.append(iname)
		gee_pt_obj.S1TS['117'].sort_index().to_csv(workpath + iname + '.csv')

	gee_pt_obj = None
	cntr = cntr + 1
	if names is not None:
		return (sm_ts_out, names_out)
	else:
		return sm_ts_out
