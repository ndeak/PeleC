function extractStatus=pelecCreateMovie(linedir, moviedir, ndens, tip1, tip2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% [SAY SOMETHING USEFUL]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Adding matlab pathways
addpath('/work2/04361/ndeak/stampede2/matlab/'); rfmlpath;
addpath('/work2/04361/ndeak/stampede2/plasmalib/matlab/');

% Check to make sure that the extracted data folder actually exists.  Warn user if it doesn't.
if ~isfolder(linedir)
    errorMessage = sprintf('Create Movie Error: linedir folder does not exist, exiting!');
    extractStatus = 'LinedirNotFound'
    return;
end

% Other variables
tref = 1.0e9;
NA = 6.0221409e23;
full_plot = 0;


% Loop over each extracted data file
filePattern = fullfile(linedir, 'fine*'); 
plotFiles = dir(filePattern);
EN_anode_tip = zeros(length(plotFiles), 2);
nE_anode_tip = zeros(length(plotFiles), 2);
Tg_anode_tip = zeros(length(plotFiles), 2);
for k = 1 : length(plotFiles)
    baseFileName = plotFiles(k).name;
    fullFileName = fullfile(plotFiles(k).folder, baseFileName);
    fprintf(1, 'Now plotting %s\n', fullFileName);

    % Get plotfile number
    fileNumStr = fullFileName(end-8:end-4);

    % Parse current time from second line of datafile
    FID = fopen(fullFileName);
    headerData = textscan(FID,'%s');
    fclose(FID);
    stringData = string(headerData{:});
    currTime = str2double(stringData(11));

    % Read in the fextracted data
    [H,lineData]=readdata(fullFileName);

    % Get number of extracted fields
    [r c] = size(lineData);
    nf = c - 1;

    % Get the reduced electric field
    ENData = (lineData(:,nf-1).^2 + lineData(:,nf).^2 + lineData(:,nf+1).^2).^0.5 * 1.0e10 / ndens;

    % Get the reduced electric field
    nEData = lineData(:,2);

    % Get the reduced electric field
    TgData = lineData(:,nf-4);

    for i = 2:length(lineData(:,1))
      if(lineData(i,1) < tip2 )
        EN_anode_tip(k,2) = ENData(i-1);
        nE_anode_tip(k,2) = nEData(i-1);
        Tg_anode_tip(k,2) = TgData(i-1);
      end
    end
    EN_anode_tip(k,1) = currTime*tref;
    nE_anode_tip(k,1) = currTime*tref;
    Tg_anode_tip(k,1) = currTime*tref;
end

EN_anodeName = strcat(moviedir, "EN_anode_data.dat");
T = table(EN_anode_tip(:,1), EN_anode_tip(:,2), 'VariableNames', { 't(ns)', 'EN_anode(Td)'} );
writetable(T, EN_anodeName,'Delimiter','\t');

nE_anodeName = strcat(moviedir, "nE_anode_data.dat");
T = table(nE_anode_tip(:,1), nE_anode_tip(:,2), 'VariableNames', { 't(ns)', 'nE_anode(cm-3)'} );
writetable(T, nE_anodeName,'Delimiter','\t');

Tg_anodeName = strcat(moviedir, "Tg_anode_data.dat");
T = table(Tg_anode_tip(:,1), Tg_anode_tip(:,2), 'VariableNames', { 't(ns)', 'Tg_anode(K)'} );
writetable(T, Tg_anodeName,'Delimiter','\t');

end
