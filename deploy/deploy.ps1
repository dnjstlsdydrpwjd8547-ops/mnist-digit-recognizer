# Zero-downtime blue/green deployment for the MNIST web app (Windows + Docker Desktop)
#
#   [client] -> http://127.0.0.1:8000 -> mnist-proxy (nginx) -> mnist-blue | mnist-green
#
# 1. 배포 전부터 고정 URL(8000)에 1초 간격으로 실제 /predict 요청을 보내는 모니터를 시작
# 2. 새 이미지를 pull 하고 비활성 색상(blue/green) 컨테이너로 실행 -> 헬스체크 + 실제 추론 확인
# 3. nginx upstream 을 새 컨테이너로 바꾸고 graceful reload (연결 끊김 없음)
# 4. 기존 컨테이너 제거, 배포 완료 후 30초 더 모니터링
# 5. 요청 로그(CSV) + 요약(GitHub Step Summary) 작성
param(
    [Parameter(Mandatory = $true)][string]$Image,
    [Parameter(Mandatory = $true)][string]$Tag,
    [int]$Port = 8000,
    [int]$AfterSeconds = 30
)

# Continue: Windows PowerShell 5.1 은 네이티브 명령의 stderr 출력을 오류로 취급하므로 종료코드로 직접 판단
$ErrorActionPreference = "Continue"
$Network = "mnist-net"
$Proxy = "mnist-proxy"
$Root = Split-Path -Parent $PSScriptRoot
$LogDir = Join-Path $Root "deploy-logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$LogCsv = Join-Path $LogDir "requests.csv"
$PhaseFile = Join-Path $LogDir "phase.txt"
$Utf8 = New-Object System.Text.UTF8Encoding($false)
$Url = "http://127.0.0.1:$Port"
$FullImage = "${Image}:${Tag}"

function Log($msg) { Write-Host ("[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $msg) }
function Docker-Quiet { & docker @args 2>$null; $global:LASTEXITCODE = 0 }
function Set-Phase($p) { [IO.File]::WriteAllText($PhaseFile, $p, $Utf8); Log "phase -> $p" }

function Write-ProxyConf($target) {
    $conf = @"
upstream mnist_app { server ${target}:8000; }
server {
    listen 80;
    location / {
        proxy_pass http://mnist_app;
        proxy_set_header Host `$host;
        proxy_connect_timeout 2s;
        proxy_read_timeout 30s;
    }
}
"@
    $tmp = Join-Path $LogDir "default.conf"
    [IO.File]::WriteAllText($tmp, ($conf -replace "`r", ""), $Utf8)
    docker cp $tmp "${Proxy}:/etc/nginx/conf.d/default.conf" | Out-Null
    docker exec $Proxy nginx -t 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "nginx config test failed" }
    docker exec $Proxy nginx -s reload 2>&1 | Out-Null
}

function Wait-Healthy($name, $timeoutSec = 90) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        $state = (docker inspect -f "{{.State.Health.Status}}" $name 2>$null)
        if ($state -eq "healthy") { return }
        Start-Sleep -Seconds 1
    }
    docker logs --tail 50 $name
    Set-Phase "stop"
    throw "container $name did not become healthy (old version keeps serving)"
}

# ---------- 0. 공통 리소스 준비 ----------
Log "image: $FullImage"
if (-not (docker network ls --format "{{.Name}}" | Where-Object { $_ -eq $Network })) {
    docker network create $Network | Out-Null
}
$active = $null
foreach ($c in @("mnist-blue", "mnist-green")) {
    if ((docker ps --format "{{.Names}}") -contains $c) { $active = $c }
}
$target = if ($active -eq "mnist-blue") { "mnist-green" } else { "mnist-blue" }
$proxyRunning = (docker ps --format "{{.Names}}") -contains $Proxy
Log "active: $(if ($active) { $active } else { '(none)' }) -> new: $target"

# ---------- 1. 모니터 ----------
function Start-Monitor($initialPhase) {
    "seq,phase,start_utc,end_utc,latency_ms,http,digit,model_version,app_version,host,error" |
        Out-File -FilePath $LogCsv -Encoding ascii
    Set-Phase $initialPhase
    $body = [IO.File]::ReadAllText((Join-Path $PSScriptRoot "sample_request.json"))
    $script:monitor = Start-Job -ArgumentList $Url, $body, $LogCsv, $PhaseFile -ScriptBlock {
        param($Url, $Body, $LogCsv, $PhaseFile)
        $seq = 0
        while ($true) {
            $phase = (Get-Content $PhaseFile -Raw).Trim()
            if ($phase -eq "stop") { break }
            $seq++
            $t0 = [DateTime]::UtcNow
            $http = 0; $digit = ""; $mv = ""; $av = ""; $hst = ""; $err = ""
            try {
                $r = Invoke-WebRequest -Uri "$Url/predict" -Method Post -Body $Body `
                    -ContentType "application/json" -UseBasicParsing -TimeoutSec 10
                $http = [int]$r.StatusCode
                $j = $r.Content | ConvertFrom-Json
                $digit = $j.digit; $mv = $j.model_version; $av = $j.app_version; $hst = $j.host
            } catch {
                $err = ($_.Exception.Message -replace "[,`r`n]", " ")
                if ($_.Exception.Response) { $http = [int]$_.Exception.Response.StatusCode }
            }
            $t1 = [DateTime]::UtcNow
            $ms = [int]($t1 - $t0).TotalMilliseconds
            "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10}" -f $seq, $phase, $t0.ToString("o"), $t1.ToString("o"),
                $ms, $http, $digit, $mv, $av, $hst, $err | Add-Content -Path $LogCsv -Encoding ascii
            Start-Sleep -Seconds 1
        }
    }
}

if ($active) {
    Start-Monitor "before"
    Start-Sleep -Seconds 5   # 배포 전 기준 요청 확보
}


# ---------- 2. 새 버전 컨테이너 기동 ----------
$deployStart = [DateTime]::UtcNow
if ($active) { Set-Phase "during" }
Log "docker pull $FullImage"
docker pull $FullImage
if ($LASTEXITCODE -ne 0) { Set-Phase "stop"; throw "docker pull failed (old version keeps serving)" }

Docker-Quiet rm -f $target | Out-Null
docker run -d --name $target --network $Network --restart unless-stopped `
    --label "mnist.tag=$Tag" $FullImage | Out-Null
Wait-Healthy $target
# 새 컨테이너에 직접 실제 추론 요청 (트래픽 전환 전 검증)
$check = Get-Content (Join-Path $PSScriptRoot "sample_request.json") -Raw |
    docker exec -i $target python -c "import sys,urllib.request;b=sys.stdin.read().encode();r=urllib.request.urlopen(urllib.request.Request('http://127.0.0.1:8000/predict',b,{'Content-Type':'application/json'}));print(r.read().decode())"
if ($LASTEXITCODE -ne 0 -or -not $check) { Set-Phase "stop"; throw "new container failed the predict check (old version keeps serving)" }
Log "new container check: $check"

# ---------- 3. 트래픽 전환 ----------
if (-not $proxyRunning) {
    Docker-Quiet rm -f $Proxy | Out-Null
    docker run -d --name $Proxy --network $Network --restart unless-stopped -p "${Port}:80" nginx:1.27-alpine | Out-Null
    Start-Sleep -Seconds 2
}
Write-ProxyConf $target
Log "traffic switched to $target"
if (-not $active) { Start-Sleep -Seconds 2; Start-Monitor "during" }   # 최초 배포: 서비스가 뜬 뒤부터 측정

# ---------- 4. 기존 버전 정리 ----------
if ($active) {
    Start-Sleep -Seconds 3      # 처리 중인 요청이 끝날 시간
    docker rm -f $active | Out-Null
    Log "removed old container $active"
}
$deployEnd = [DateTime]::UtcNow
Set-Phase "after"
Start-Sleep -Seconds $AfterSeconds
Set-Phase "stop"
Wait-Job $monitor -Timeout 30 | Out-Null
Remove-Job $monitor -Force

# ---------- 5. 요약 ----------
$rows = Import-Csv $LogCsv
$total = $rows.Count
$ok = @($rows | Where-Object { $_.http -eq "200" -and $_.digit -ne "" }).Count
$fail = $total - $ok
$maxGap = 0.0
for ($i = 1; $i -lt $total; $i++) {
    $gap = ([DateTime]$rows[$i].start_utc - [DateTime]$rows[$i - 1].end_utc).TotalSeconds
    if ($gap -gt $maxGap) { $maxGap = $gap }
}
$md = New-Object System.Text.StringBuilder
[void]$md.AppendLine("## MNIST zero-downtime deployment verification")
[void]$md.AppendLine("")
[void]$md.AppendLine("- **Result:** $(if ($fail -eq 0) { 'SUCCESS - no failed requests' } else { "FAILED requests: $fail" })")
[void]$md.AppendLine("- **Stable URL:** $Url  |  **Image:** ``$FullImage``")
[void]$md.AppendLine("- **Switch:** $(if ($active) { $active } else { '(first deploy)' }) -> $target")
[void]$md.AppendLine("- **Deploy window (UTC):** $($deployStart.ToString('HH:mm:ss.fff')) - $($deployEnd.ToString('HH:mm:ss.fff'))")
[void]$md.AppendLine("- **API requests:** $total total / $ok success / $fail failed  |  interval 1s after each response  |  largest gap between requests: $([math]::Round($maxGap,2)) s")
[void]$md.AppendLine("")
[void]$md.AppendLine("| Stage | Requests (ok/total) | Avg latency (ms) | Model version | App version (git sha) | Container | Result |")
[void]$md.AppendLine("|---|---|---|---|---|---|---|")
foreach ($ph in @("before", "during", "after")) {
    $r = @($rows | Where-Object { $_.phase -eq $ph })
    if ($r.Count -eq 0) { [void]$md.AppendLine("| $ph | 0/0 | - | - | - | - | - |"); continue }
    $o = @($r | Where-Object { $_.http -eq "200" -and $_.digit -ne "" })
    $avg = [math]::Round((($r | Measure-Object -Property latency_ms -Average).Average), 1)
    $mv = (($r | Where-Object { $_.model_version } | Select-Object -ExpandProperty model_version -Unique) -join " -> ")
    $av = (($r | Where-Object { $_.app_version } | ForEach-Object { $_.app_version.Substring(0, [math]::Min(7, $_.app_version.Length)) } | Select-Object -Unique) -join " -> ")
    $hs = (($r | Where-Object { $_.host } | Select-Object -ExpandProperty host -Unique) -join " -> ")
    $digits = (($o | Select-Object -ExpandProperty digit -Unique) -join ",")
    [void]$md.AppendLine("| $ph | $($o.Count)/$($r.Count) | $avg | $mv | $av | $hs | digit=$digits |")
}
[void]$md.AppendLine("")
[void]$md.AppendLine("Full request log: artifact ``deploy-logs`` (requests.csv)")
$summary = $md.ToString()
[IO.File]::WriteAllText((Join-Path $LogDir "summary.md"), $summary, $Utf8)
if ($env:GITHUB_STEP_SUMMARY) { [IO.File]::AppendAllText($env:GITHUB_STEP_SUMMARY, $summary, $Utf8) }
Write-Host $summary
if ($fail -gt 0) { exit 1 }
