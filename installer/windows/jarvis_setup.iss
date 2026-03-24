; Jarvis Inno Setup Script
; Builds a Windows installer from the PyInstaller onedir output.
;
; Usage:
;   iscc installer\windows\jarvis_setup.iss
;
; Expects the PyInstaller onedir output at dist\Jarvis\

#define MyAppName "Jarvis"
#define MyAppExeName "Jarvis.exe"
#define MyAppPublisher ""

; VC++ Redistributable download URL (VS 2015-2022 x64)
#define VCRedistURL "https://aka.ms/vs/17/release/vc_redist.x64.exe"

[Setup]
AppId={{B8A3D6F1-7C42-4E5A-9D12-3F8E6A1B5C90}
AppName={#MyAppName}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\..\dist
OutputBaseFilename=Jarvis-Setup-x64
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
ArchitecturesAllowed=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}
PrivilegesRequired=admin
SetupIconFile=..\..\src\desktop_app\desktop_assets\icon_idle.ico

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Bundle the entire PyInstaller onedir output
Source: "..\..\dist\Jarvis\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
; Install VC++ Redistributable silently if missing
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/quiet /norestart"; StatusMsg: "Installing Visual C++ Redistributable..."; Flags: waituntilterminated; Check: VCRedistNeeded
; Launch the application after installation
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"

[Code]
// Check whether the VC++ 2015-2022 runtime is already installed
function VCRedistNeeded: Boolean;
var
  Version: String;
begin
  // Check for VC++ 2015-2022 x64 runtime via registry
  Result := True;
  if RegQueryStringValue(HKLM, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Version', Version) then
  begin
    // Runtime is installed
    Result := False;
  end;
end;

// Download VC++ Redistributable if needed
procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssInstall then
  begin
    if VCRedistNeeded then
    begin
      // Download vc_redist.x64.exe from Microsoft
      DownloadTemporaryFile('{#VCRedistURL}', 'vc_redist.x64.exe', '', nil);
    end;
  end;
end;

// After installation, clean up the old exe if the installer was launched
// from a legacy location (e.g. old updater placed it at a custom path).
// The installer can't delete itself while running, so we schedule a
// cmd /c del command that retries until the file is unlocked.
procedure DeinitializeSetup;
var
  InstallerPath, InstalledDir: String;
begin
  InstallerPath := ExpandConstant('{srcexe}');
  InstalledDir := ExpandConstant('{app}');
  // Only clean up if the installer is NOT inside the installation directory
  // (i.e. it was placed somewhere else by the old updater)
  if Pos(Lowercase(InstalledDir), Lowercase(InstallerPath)) = 0 then
  begin
    Log('Scheduling cleanup of old installer at: ' + InstallerPath);
    Exec('cmd.exe',
      '/c ping -n 3 127.0.0.1 >nul & del /f "' + InstallerPath + '"',
      '', SW_HIDE, ewNoWait, 0);
  end;
end;

