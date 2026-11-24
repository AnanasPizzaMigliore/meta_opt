package org.example.questcap;

import android.app.Activity;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.media.Image;
import android.media.AudioManager;
import android.media.ImageReader;
import android.media.ToneGenerator;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Range;
import android.view.Gravity;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

import java.io.File;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Locale;

/**
 * Dataset capture for the expiration-date study. Shows a product number and a clip letter; "Record" saves
 * 12 s of the left passthrough camera (id 50, 1728x1728) at up to 10 fps as JPEG plus a frames.tsv with sensor
 * timestamps, to files/captures/p<NNN>/<clip>/ in the app's external files dir. "Next" / "Prev" change the
 * product number, "Clip" cycles a (bent arm) / b (close) / d (straight arm) / c (no date); s = size reference. A beep marks the start and a double beep the end of a recording.
 *
 * Remote control (no controllers needed):
 *   am start -n org.example.questcap/.MainActivity --ei product 3 --es clip a --ez record true
 * selects the product and clip and starts recording after a 2 s warning beep.
 */
public class MainActivity extends Activity {
    private static final String CAMERA_ID = "50";
    private static final int W = 1728, H = 1728, SECONDS = 12, KEEP_EVERY = 3;   // 30 fps sensor -> 10 fps saved

    private TextView status, number;
    private int product = 1;
    private String clip = "a";
    private volatile long startMs = 0;
    private ToneGenerator tone;
    private HandlerThread thread;
    private Handler bg;
    private CameraDevice camera;
    private ImageReader reader;
    private volatile boolean recording = false;
    private volatile int seen = 0, saved = 0;
    private File dir;
    private StringBuilder tsv;

    @Override
    protected void onCreate(Bundle b) {
        super.onCreate(b);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setGravity(Gravity.CENTER);
        root.setBackgroundColor(Color.rgb(24, 24, 24));
        number = new TextView(this);
        number.setTextSize(64);
        number.setTextColor(Color.WHITE);
        number.setGravity(Gravity.CENTER);
        status = new TextView(this);
        status.setTextSize(22);
        status.setTextColor(Color.LTGRAY);
        status.setGravity(Gravity.CENTER);
        LinearLayout row = new LinearLayout(this);
        row.setGravity(Gravity.CENTER);
        Button prev = button("Prev"), rec = button("Record 12 s"), cl = button("Clip"), next = button("Next");
        row.addView(prev); row.addView(rec); row.addView(cl); row.addView(next);
        root.addView(number); root.addView(status); root.addView(row);
        setContentView(root);
        prev.setOnClickListener(v -> { if (!recording && product > 1) { product--; show(); } });
        next.setOnClickListener(v -> { if (!recording) { product++; show(); } });
        rec.setOnClickListener(v -> startRecording());
        cl.setOnClickListener(v -> { if (!recording) { clip = clip.equals("a") ? "b" : clip.equals("b") ? "d" : clip.equals("d") ? "c" : "a"; show(); } });
        try { tone = new ToneGenerator(AudioManager.STREAM_MUSIC, 90); } catch (Exception e) { tone = null; }
        thread = new HandlerThread("cam");
        thread.start();
        bg = new Handler(thread.getLooper());
        show();
        String[] perms = {"android.permission.CAMERA", "horizonos.permission.HEADSET_CAMERA"};
        if (checkSelfPermission(perms[0]) != PackageManager.PERMISSION_GRANTED
                || checkSelfPermission(perms[1]) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(perms, 1);
        } else {
            openCamera();
        }
        remote(getIntent());
    }

    @Override
    protected void onNewIntent(Intent i) {
        super.onNewIntent(i);
        remote(i);
    }

    /** Product / clip selection and a delayed Record from intent extras (see the class comment). */
    private void remote(Intent i) {
        if (i == null || recording) return;
        if (i.hasExtra("product")) product = Math.max(1, i.getIntExtra("product", product));
        String c = i.getStringExtra("clip");
        if (c != null && c.matches("[a-z]")) clip = c;
        show();
        if (i.getBooleanExtra("record", false)) {
            beep(ToneGenerator.TONE_PROP_BEEP);
            status.setText("recording starts in 2 s ...");
            status.postDelayed(this::startRecording, 2000);
        }
    }

    private void beep(int t) {
        try { if (tone != null) tone.startTone(t, 300); } catch (Exception e) { /* sound is optional */ }
    }

    private Button button(String text) {
        Button bt = new Button(this);
        bt.setText(text);
        bt.setTextSize(26);
        bt.setAllCaps(false);
        return bt;
    }

    private void show() {
        File d = clipDir(product, clip);
        String[] f = d.list();
        int n = f == null ? 0 : f.length;
        number.setText(String.format(Locale.US, "Product %03d\nclip %s - %s", product, clip, hint(clip)));
        status.setText(n > 0 ? "already recorded (" + n + " files) - Record again overwrites" : "not recorded yet");
    }

    /** What the wearer should do for this clip letter. */
    private static String hint(String c) {
        switch (c) {
            case "a": return "bent arm";
            case "b": return "close";
            case "d": return "straight arm";
            case "c": return "side with NO date";
            case "s": return "on the A4 sheet";
            default:  return "";
        }
    }

    private File clipDir(int p, String c) {
        return new File(getExternalFilesDir(null), String.format(Locale.US, "captures/p%03d/%s", p, c));
    }

    @Override
    public void onRequestPermissionsResult(int code, String[] p, int[] r) {
        if (r.length == 2 && r[0] == PackageManager.PERMISSION_GRANTED && r[1] == PackageManager.PERMISSION_GRANTED) openCamera();
        else status.setText("camera permission denied");
    }

    private void openCamera() {
        try {
            CameraManager cm = (CameraManager) getSystemService(CAMERA_SERVICE);
            reader = ImageReader.newInstance(W, H, ImageFormat.YUV_420_888, 4);
            reader.setOnImageAvailableListener(this::onImage, bg);
            cm.openCamera(CAMERA_ID, new CameraDevice.StateCallback() {
                @Override public void onOpened(CameraDevice c) { camera = c; startPreview(); }
                @Override public void onDisconnected(CameraDevice c) { c.close(); camera = null; }
                @Override public void onError(CameraDevice c, int e) { c.close(); camera = null; ui("camera error " + e); }
            }, bg);
        } catch (Exception e) {
            ui("cannot open camera: " + e);
        }
    }

    private void startPreview() {
        try {
            camera.createCaptureSession(Arrays.asList(reader.getSurface()), new CameraCaptureSession.StateCallback() {
                @Override public void onConfigured(CameraCaptureSession s) {
                    try {
                        CaptureRequest.Builder rb = camera.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
                        rb.addTarget(reader.getSurface());
                        rb.set(CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE, new Range<>(30, 30));
                        s.setRepeatingRequest(rb.build(), null, bg);
                        ui("camera ready - hold the product, then press Record");
                    } catch (Exception e) { ui("capture error: " + e); }
                }
                @Override public void onConfigureFailed(CameraCaptureSession s) { ui("camera configuration failed"); }
            }, bg);
        } catch (Exception e) {
            ui("session error: " + e);
        }
    }

    private void startRecording() {
        if (camera == null || recording) { if (camera == null) ui("camera not ready"); return; }
        dir = clipDir(product, clip);
        dir.mkdirs();
        File[] old = dir.listFiles();
        if (old != null) for (File f : old) f.delete();
        tsv = new StringBuilder("file\tsensor_timestamp_ns\twall_ms\n");
        seen = 0;
        saved = 0;
        startMs = System.currentTimeMillis();
        recording = true;
        beep(ToneGenerator.TONE_PROP_BEEP);
        ui("RECORDING product " + product + " clip " + clip + " (" + SECONDS + " s) ...");
    }

    private void onImage(ImageReader r) {
        Image img = r.acquireLatestImage();
        if (img == null) return;
        try {
            if (!recording) return;
            if (seen++ % KEEP_EVERY != 0) return;
            String name = String.format(Locale.US, "f%03d.jpg", saved);
            byte[] nv21 = toNv21(img);
            try (FileOutputStream out = new FileOutputStream(new File(dir, name))) {
                new YuvImage(nv21, ImageFormat.NV21, W, H, null).compressToJpeg(new Rect(0, 0, W, H), 95, out);
            }
            tsv.append(name).append('\t').append(img.getTimestamp()).append('\t').append(System.currentTimeMillis()).append('\n');
            saved++;
            if (System.currentTimeMillis() - startMs >= SECONDS * 1000L) {
                recording = false;
                try (FileOutputStream out = new FileOutputStream(new File(dir, "frames.tsv"))) {
                    out.write(tsv.toString().getBytes());
                }
                beep(ToneGenerator.TONE_PROP_BEEP2);
                ui("saved " + saved + " frames for product " + product + " clip " + clip);
            }
        } catch (Exception e) {
            recording = false;
            ui("save error: " + e);
        } finally {
            img.close();
        }
    }

    private static byte[] toNv21(Image img) {
        Image.Plane y = img.getPlanes()[0], u = img.getPlanes()[1], v = img.getPlanes()[2];
        int w = img.getWidth(), h = img.getHeight();
        byte[] out = new byte[w * h * 3 / 2];
        ByteBuffer yb = y.getBuffer();
        int ys = y.getRowStride(), pos = 0;
        for (int row = 0; row < h; row++) {
            yb.position(row * ys);
            yb.get(out, pos, w);
            pos += w;
        }
        ByteBuffer ub = u.getBuffer(), vb = v.getBuffer();
        int ucs = u.getRowStride(), ups = u.getPixelStride(), vcs = v.getRowStride(), vps = v.getPixelStride();
        for (int row = 0; row < h / 2; row++) {
            for (int col = 0; col < w / 2; col++) {
                out[pos++] = vb.get(row * vcs + col * vps);
                out[pos++] = ub.get(row * ucs + col * ups);
            }
        }
        return out;
    }

    private void ui(String s) {
        runOnUiThread(() -> { status.setText(s); if (s.startsWith("saved")) show(); status.setText(s); });
    }

    @Override
    protected void onDestroy() {
        if (camera != null) camera.close();
        if (reader != null) reader.close();
        if (tone != null) tone.release();
        thread.quitSafely();
        super.onDestroy();
    }
}
