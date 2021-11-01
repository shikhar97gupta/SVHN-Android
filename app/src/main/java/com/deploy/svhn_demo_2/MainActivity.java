package com.deploy.svhn_demo_2;


import android.app.Activity;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.model.Model;

import java.io.IOException;
import java.nio.MappedByteBuffer;

public class MainActivity extends Activity {

    ImageView imageView;
    TextView textView;
    Button button;
    private static final int PICK_IMAGE = 100;
    Uri imageUri;

    @Override

    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = (ImageView)findViewById(R.id.imageView);
        textView = (TextView)findViewById(R.id.textView);
        button = (Button)findViewById(R.id.buttonLoadPicture);
        button.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                openGallery();
            }
        });
    }

    private void openGallery() {
        Intent gallery = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.INTERNAL_CONTENT_URI);
        startActivityForResult(gallery, PICK_IMAGE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data){
        super.onActivityResult(requestCode, resultCode, data);

        if (resultCode == RESULT_OK && requestCode == PICK_IMAGE){
            imageUri = data.getData();
            imageView.setImageURI(imageUri);

            Bitmap bitmap = null;

            try {
                bitmap = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
                Bitmap bmp = bitmap.copy(Bitmap.Config.ARGB_8888,true);
                TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
                tensorImage.load(bmp);
                TensorBuffer probabilityBuffer =
                        TensorBuffer.createFixedSize(new int[]{10}, DataType.FLOAT32);

                try{
                    MappedByteBuffer tfliteModel
                            = FileUtil.loadMappedFile(this,
                            "svhn.tflite");
                    Interpreter tflite = new Interpreter(tfliteModel);
                    if(null != tflite) {
                        tflite.run(tensorImage.getBuffer(), probabilityBuffer.getBuffer());
                    }
                    float[] tmp = probabilityBuffer.getFloatArray();
                    String s = String.format("%f %f %f %f %f %f %f %f %f %f", tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5],
                                                                                tmp[6], tmp[7], tmp[8], tmp[9]);
                    textView.setText(s);
                }
                catch (IOException e){
                    Log.e("tfliteSupport", "Error reading model", e);
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }


    private String img_load(){
        return "Loaded";
    }
}