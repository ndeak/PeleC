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

% Set up movie variables
MovieName = strcat(moviedir, "TimeSeriesFixedAxis.avi");
MovieNameLin = strcat(moviedir, "TimeSeriesFixedAxisLinear.avi");
MovieNameCathode = strcat(moviedir, "TimeSeriesCathodeRegionLog.avi");
vi = VideoWriter(MovieName);
vi.FrameRate = 1;
vil = VideoWriter(MovieNameLin);
vil.FrameRate = 1;
vic = VideoWriter(MovieNameCathode);
vic.FrameRate = 1;

% Loop over each extracted data file
filePattern = fullfile(linedir, '*.dat'); 
plotFiles = dir(filePattern);
headLocation = zeros(length(plotFiles), 2);
maxETS = zeros(length(plotFiles), 2);
maxENTS = zeros(length(plotFiles), 2);
sheathLength = zeros(length(plotFiles), 2);
voltageDrop = zeros(length(plotFiles), 2);
open(vi);
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

    % Set up linetypes
    if full_plot == 1
      linS = {'-r','--g',':b', '-k', '--b', ':r', '-g'};
    else
      linS = {'-r','-g','-b','-m'};
    end

    % Hard-coding voltage calculation for now...
    currVoltage = (1.0 - abs(currTime - 1.0e-8)/1.0e-8)*40.0;

    % Set up axes
    f1 = figure;
    xlim([tip1 tip2])
    ylim([1.0e2 1.0e16])                    % FIXME: Hard-coding extends for now...
    xlabel('x [cm]')
    set(gca, 'YScale', 'log')
    ylabel('n [cm-3]')
    yyaxis right
    ylim([0.0 3500])                    % FIXME: Hard-coding extends for now...
    ylabel('E/N [Td]')
    title(['n (cm-3) and E/N [Td], t = ',num2str(currTime*tref),' ns, Va = ', num2str(currVoltage), ' kV, tag = ', fileNumStr])
    yyaxis left
    hold on;

    % Plot the number densities
    if full_plot == 1
      for n = 1:nf-4
          semilogy(lineData(:,1), abs(lineData(:,n+1)),linS{n});
      end
    else
      pions = abs(lineData(:,3)) + abs(lineData(:,4)) + abs(lineData(:,5)) + abs(lineData(:,6)) + abs(lineData(:,7));
      nions = abs(lineData(:,8));
      charge = abs(pions - nions - abs(lineData(:,2)));
      semilogy(lineData(:,1), abs(lineData(:,2)), linS{1});
      semilogy(lineData(:,1), pions, linS{2});
      semilogy(lineData(:,1), nions, linS{3});
      semilogy(lineData(:,1), charge, linS{4});
    end


    % Plot the reduced electric field
    yyaxis right
    ENData = (lineData(:,nf-1).^2 + lineData(:,nf).^2 + lineData(:,nf+1).^2).^0.5 * 1.0e10 / ndens;
    plot(lineData(:,1), ENData, '-k');

    if full_plot == 1
        legend('n(E)', 'n(N2+)', 'n(N4+)', 'n(O2+)', 'n(O4+)', 'n(O2pN2)', 'n(O2-)', 'E/N','Location','south');
    else
        legend('n(E)', 'n(+)', 'n(-)', 'rho_c', 'E/N','Location','south');
    end

    delete(findall(gcf,'Type','hggroup'));
    frame = getframe(gcf);
    writeVideo(vi,frame);
    hold off;
    close(f1)

    % Find the maximum reduced electric field to approximate location of streamer head
    % Skip the first few cells by domain boundary to avoid large gradients near the electrodes
    % Also find the maximum electric field (abs) value at each point in time
    bottombuff = tip1 + (tip2-tip1)/20.0;
    topbuff = tip2;
    maxEN = 0.0;    
    maxETSval = 0.0;
    maxLoc = 0.0;
    for i = 1:length(lineData(:,1))
      if (lineData(i,1) > bottombuff) && (lineData(i,1) <= topbuff) && (ENData(i) > maxEN)
        maxEN = ENData(i);
        maxLoc = lineData(i,1);
        maxETSval = ENData(i) * 1.0e-17 * ndens * 1.0e-3;   % Convert Td -> V-cm2 -> V/cm -> kV/cm
      end
    end    
    headLocation(k,1) = currTime*tref;
    headLocation(k,2) = 10.0*abs(tip2 - maxLoc);
    maxETS(k,1) = currTime*tref;
    maxETS(k,2) = maxETSval;
    maxENTS(k,1) = currTime*tref;
    maxENTS(k,2) = maxEN;

    % Find the length of the cathode sheath defined as the first point, starting from 
    % the cathode, where the electron number density surpasses half the postive ion density
    % We also take the voltage drop to be the value of the voltage at the sheath location
    % (note that this assumes a grounded cathode)
    for i = 1:length(lineData(:,1))
      pos_ions = abs(lineData(i,3)) + abs(lineData(i,4)) + abs(lineData(i,5)) + abs(lineData(i,6)) + abs(lineData(i,7));
      if(lineData(i,1) > tip1 && lineData(i,1) < tip2 && lineData(i,2) > pos_ions / 2.0)
        sheathLocation(k,2) = (lineData(i,1) - tip1)*10.0;
        voltageDrop(k,2) = lineData(i,nf-2) * 1.0e-10;    % Converting to kV
        break
      end
    end
    sheathLocation(k,1) = currTime*tref;
    voltageDrop(k,1) = currTime*tref;
end
close(vi);

% Plot the location of the streamer head as a function of time
f1 = figure;
plotName = strcat(moviedir, "StreamerHeadLocation.png");
xlabel('t [ns]')
ylabel('Streamer distance from anode [mm]')
hold on;
plot(headLocation(:,1), headLocation(:,2))
saveas(gcf, plotName)
HeadLocName = strcat(moviedir, "head_location.dat");
T = table(headLocation(:,1), headLocation(:,2), 'VariableNames', { 't(ns)', 'anode_dist(mm)'} );
writetable(T, HeadLocName,'Delimiter','\t');

% Plot the maximum electric field magnitude as a function of time
f1 = figure;
plotName = strcat(moviedir, "MaxETimeSeries.png");
xlabel('t [ns]')
ylabel('E max [kV/cm]')
hold on;
plot(maxETS(:,1), maxETS(:,2))
saveas(gcf, plotName)
EmaxName = strcat(moviedir, "E_max.dat");
T = table(maxETS(:,1), maxETS(:,2), 'VariableNames', { 't(ns)', 'E_max(kV/cm)'} );
writetable(T, EmaxName,'Delimiter','\t');

% Plot the maximum reduced electric field as a function of time
f1 = figure;
plotName = strcat(moviedir, "MaxENTimeSeries.png");
xlabel('t [ns]')
ylabel('E/N max [Td]')
hold on;
plot(maxENTS(:,1), maxENTS(:,2))
saveas(gcf, plotName)
ENmaxName = strcat(moviedir, "EN_max.dat");
T = table(maxENTS(:,1), maxENTS(:,2), 'VariableNames', { 't(ns)', 'EN_max(Td)'} );
writetable(T, ENmaxName,'Delimiter','\t');

% Plot the cathode sheath length and voltage drop as a function of time
f1 = figure;
plotName = strcat(moviedir, "CathodeSheathTimeSeries.png");
xlabel('t [ns]')
ylabel('Sheath Length [mm]')
yyaxis right
ylabel('Cathode Voltage Drop [kV]')
hold on;
yyaxis left
plot(sheathLocation(:,1), sheathLocation(:,2))
yyaxis right
plot(voltageDrop(:,1), voltageDrop(:,2))
legend('sheath', 'drop','Location','south');
saveas(gcf, plotName)
CathVoltName = strcat(moviedir, "cathode_sheath_voltage.dat");
T = table(sheathLocation(:,1), sheathLocation(:,2), voltageDrop(:,2), 'VariableNames', { 't(ns)', 'sheath_length(mm)', 'voltage_drop(kV)'} );
writetable(T, CathVoltName,'Delimiter','\t');

% Plot linear scale if needed
open(vil);
for k = 1 : length(plotFiles)
    baseFileName = plotFiles(k).name;
    fullFileName = fullfile(plotFiles(k).folder, baseFileName);
    fprintf(1, 'Linear plot! Now plotting %s\n', fullFileName);

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

    % Set up linetypes
    if full_plot == 1
      linS = {'-r','--g',':b', '-k', '--b', ':r', '-g'};
    else
      linS = {'-r','-g','-b','-m'};
    end

    % Hard-coding voltage calculation for now...
    currVoltage = (1.0 - abs(currTime - 1.0e-8)/1.0e-8)*40.0;

    % Set up axes
    f1 = figure;
    % xlim([1.46142578125 2.53857421875])     % FIXME: Hard-coding extends for now...
    xlim([tip1 tip2])
    xlabel('x [cm]')
    ylabel('n [cm-3]')
    yyaxis right
    ylim([0.0 3500])                    % FIXME: Hard-coding extends for now...
    ylabel('E/N [Td]')
    title(['LINEAR n (cm-3) and E/N [Td], t = ',num2str(currTime*tref),' ns, Va = ', num2str(currVoltage), ' kV, tag = ', fileNumStr])
    yyaxis left
    hold on;

    % Plot the number densities
    if full_plot == 1
      for n = 1:nf-4
          plot(lineData(:,1), abs(lineData(:,n+1)),linS{n});
      end
    else
      pions = abs(lineData(:,3)) + abs(lineData(:,4)) + abs(lineData(:,5)) + abs(lineData(:,6)) + abs(lineData(:,7));
      nions = abs(lineData(:,8));
      charge = abs(pions - nions - abs(lineData(:,2)));
      plot(lineData(:,1), abs(lineData(:,2)), linS{1});
      plot(lineData(:,1), pions, linS{2});
      plot(lineData(:,1), nions, linS{3});
      plot(lineData(:,1), charge, linS{4});
    end


    % Plot the reduced electric field
    yyaxis right
    ENData = (lineData(:,nf-1).^2 + lineData(:,nf).^2 + lineData(:,nf+1).^2).^0.5 * 1.0e10 / ndens;
    plot(lineData(:,1), ENData, '-k');

    if full_plot == 1
        legend('n(E)', 'n(N2+)', 'n(N4+)', 'n(O2+)', 'n(O4+)', 'n(O2pN2)', 'n(O2-)', 'E/N','Location','south');
    else
        legend('n(E)', 'n(+)', 'n(-)', 'rho_c', 'E/N','Location','south');
    end

    delete(findall(gcf,'Type','hggroup'));
    frame = getframe(gcf);
    writeVideo(vil,frame);
    hold off;
    close(f1)
end
close(vil);

% Plot cathode region
open(vic);
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

    % Set up linetypes
    if full_plot == 1
      linS = {'-r','--g',':b', '-k', '--b', ':r', '-g'};
    else
      linS = {'-r','-g','-b','-m'};
    end

    % Hard-coding voltage calculation for now...
    currVoltage = (1.0 - abs(currTime - 1.0e-8)/1.0e-8)*40.0;

    % Set up axes
    f1 = figure;
    % xlim([1.46142578125 2.53857421875])     % FIXME: Hard-coding extends for now...
    cath_ext = tip1*1.1;
    xlim([tip1 cath_ext])
    ylim([1.0e2 1.0e16])                    % FIXME: Hard-coding extends for now...
    xlabel('x [cm]')
    set(gca, 'YScale', 'log')
    ylabel('n [cm-3]')
    yyaxis right
    ylim([0.0 3500])                    % FIXME: Hard-coding extends for now...
    ylabel('E/N [Td]')
    title(['n (cm-3) and E/N [Td], t = ',num2str(currTime*tref),' ns, Va = ', num2str(currVoltage), ' kV, tag = ', fileNumStr])
    yyaxis left
    hold on;

    % Plot the number densities
    if full_plot == 1
      for n = 1:nf-4
          semilogy(lineData(:,1), abs(lineData(:,n+1)),linS{n});
      end
    else
      pions = abs(lineData(:,3)) + abs(lineData(:,4)) + abs(lineData(:,5)) + abs(lineData(:,6)) + abs(lineData(:,7));
      nions = abs(lineData(:,8));
      charge = abs(pions - nions - abs(lineData(:,2)));
      semilogy(lineData(:,1), abs(lineData(:,2)), linS{1});
      semilogy(lineData(:,1), pions, linS{2});
      semilogy(lineData(:,1), nions, linS{3});
      semilogy(lineData(:,1), charge, linS{4});
    end


    % Plot the reduced electric field
    yyaxis right
    ENData = (lineData(:,nf-1).^2 + lineData(:,nf).^2 + lineData(:,nf+1).^2).^0.5 * 1.0e10 / ndens;
    plot(lineData(:,1), ENData, '-k');

    if full_plot == 1
        legend('n(E)', 'n(N2+)', 'n(N4+)', 'n(O2+)', 'n(O4+)', 'n(O2pN2)', 'n(O2-)', 'E/N','Location','south');
    else
        legend('n(E)', 'n(+)', 'n(-)', 'rho_c', 'E/N','Location','south');
    end

    delete(findall(gcf,'Type','hggroup'));
    frame = getframe(gcf);
    writeVideo(vic,frame);
    hold off;
    close(f1)
end
close(vic);

end
