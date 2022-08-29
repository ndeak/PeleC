function extractStatus=pelecLineExtract(vars, d, xc, yc, zc, datadir, linedir, extractdir);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% [SAY SOMETHING USEFUL]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lineFileBase = 'radial_cathode_Data_';

% Check to make sure the data folder actually exists.  Warn user if it doesn't.
if ~isfolder(datadir)
    errorMessage = sprintf('Error: The datadir folder does not exist, exiting!');
    extractStatus = 'FolderNotFound'
    return;
end

% Get all plot files in the datadir
% NOTE: It is assumed each plotfile has prefix 'plt'
% NOTE: assumed that the PeleC plotfile number is comprised of 5 digits for now..
%       (make dynamic later, maybe?)
filePattern = fullfile(datadir, 'plt*'); 
plotFiles = dir(filePattern);

numFiles = length(plotFiles)

% Get a list of all files in the folder
for k = 1 : length(plotFiles)
    baseFileName = plotFiles(k).name;
    fullFileName = fullfile(plotFiles(k).folder, baseFileName);
    fprintf(1, 'Now extracting %s\n', fullFileName);

    % Get plotfile number
    fileNumStr = fullFileName(end-5:end);
    letterar = isletter(fileNumStr);
    if letterar(1) == 0
      continue
    end
    if letterar(1) == 1
      fileNumStr = fullFileName(end-4:end);
    end

    % Construct extracted data file names
    ExtractFile = strcat(linedir, lineFileBase, fileNumStr, ".dat");

    % Construct the data extraction command
    % (Assumed post-processing on a Stampede2 skylake node with 48 processors per core - make this user-specified?)
    ExtractCmd = strcat("mpirun -np 111 /", extractdir, "/fextract.gnu.MPI.ex -s /", ExtractFile, strcat(" -c 4 -f 5 -d ", num2str(d)), strcat(" -x ", num2str(xc)), strcat(" -y ", num2str(yc)), strcat(" -z ", num2str(zc)), strcat(" -v ", vars), " -e /", fullFileName);

    % Extract data
    status = system(ExtractCmd);
end



end
