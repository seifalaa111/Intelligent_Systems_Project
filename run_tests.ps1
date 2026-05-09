$tests = @(
    @{in="i have an idea";                                                       expect="chat"},
    @{in="hello";                                                                expect="chat"},
    @{in="hey";                                                                  expect="chat"},
    @{in="give me feedback";                                                     expect="chat"},
    @{in="i want to build something";                                            expect="chat"},
    @{in="i am thinking about a startup";                                        expect="chat"},
    @{in="what do you think";                                                    expect="chat"},
    @{in="A telemedicine platform for patients in Dubai to book doctors online"; expect="analysis"},
    @{in="An invoice financing app for Egyptian SMEs to get paid faster";        expect="analysis"},
    @{in="SaaS CRM for small businesses in Saudi Arabia to manage sales";        expect="analysis"}
)

$pass = 0
$fail = 0

foreach ($t in $tests) {
    $body = '{"context":{},"messages":[{"role":"user","content":"' + $t.in + '"}]}'
    try {
        $r = Invoke-RestMethod -Method POST -Uri "http://localhost:8000/interact" -ContentType "application/json" -Body $body -TimeoutSec 30
        $got = $r.type
    } catch {
        $got = "TIMEOUT"
    }

    $ok = $got -eq $t.expect
    if ($ok) { $pass++ } else { $fail++ }
    $icon = if ($ok) { "PASS" } else { "FAIL" }
    Write-Host "[$icon] exp=$($t.expect.PadRight(10)) got=$($got.PadRight(10)) | $($t.in)"
}

Write-Host ""
Write-Host "=== $pass/$($tests.Count) PASSED ==="
