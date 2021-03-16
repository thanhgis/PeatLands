from pysmm.GEE_wrappers import GEE_extent
import numpy as np
import os
import math
import datetime as dt

def get_map(minlon, minlat, maxlon, maxlat, outpath, sampling=50, year=None, month=None, day=None, tracknr=None, overwrite=False, start=None, stop=None):

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
